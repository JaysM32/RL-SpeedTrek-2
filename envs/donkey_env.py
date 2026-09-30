"""gym-donkeycar environment wrapper.

Provides:
  - make_raw_donkey_env(...)   raw sim env, image observations, our reward shaping applied
  - make_vae_env(...)          same, but observations are VAE latent vectors (for SAC)
  - a CLI smoke test:  python -m envs.donkey_env --sim-path /path/to/donkey_sim --steps 200

gym-donkeycar's exact API (old-style `gym` 4-tuple step vs. `gymnasium` 5-tuple, and
reset() returning `obs` vs `(obs, info)`) has varied across versions. The small `_compat_*`
helpers below normalize both to the gymnasium 5-tuple/`(obs, info)` convention so the rest
of this codebase doesn't need to care which one you have installed. If your installed
version differs enough to break this, the fix is almost always in these two helpers.
"""

import argparse
import uuid

import cv2
import numpy as np
import torch
from gymnasium import Env, ObservationWrapper, RewardWrapper, spaces
from gymnasium.wrappers import TimeLimit

from utils.reward import RewardConfig, compute_reward
from vae.model import VAE


def _compat_reset(env, **kwargs):
    result = env.reset(**kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _compat_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        return result
    obs, reward, done, info = result
    return obs, reward, done, False, info


def _make_sim_conf(sim_path: str, host: str, port: int, frame_skip: int = 1, cam_resolution=(80, 160, 3),
                    max_cte: float = 8.0) -> dict:
    return {
        "exe_path": sim_path,
        "host": host,
        "port": port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "rl-agent",
        "font_size": 60,
        "racer_name": "RL",
        "country": "AU",
        "bio": "training",
        "guid": str(uuid.uuid4()),
        # This is the SIMULATOR's own built-in game-over threshold (donkey_sim.py checks
        # abs(cte) > this itself and sets `done`), separate from RewardConfig.max_cte, which
        # is our own Python-side check in compute_reward(). They used to be two independently
        # hardcoded numbers (8 here vs. whatever was in configs/sac_vae.yaml) -- now this one
        # is derived from RewardConfig.max_cte by make_raw_donkey_env() below, so there's only
        # one number to tune instead of two silently-different ones.
        "max_cte": max_cte,
        # gym-donkeycar reads these two straight off the conf dict (see supply_defaults()
        # in its donkey_env.py) -- without setting them here they silently fall back to
        # frame_skip=1 and cam_resolution=(120,160,3), no matter what configs/sac_vae.yaml says.
        "frame_skip": frame_skip,
        "cam_resolution": cam_resolution,
    }


class RewardShapingWrapper(RewardWrapper):
    """Replaces the sim's default reward with utils.reward.compute_reward, and ends the
    episode on crash/off-track (gym-donkeycar's own `done` signal is sometimes lenient)."""

    def __init__(self, env, reward_cfg: RewardConfig | None = None):
        super().__init__(env)
        self.reward_cfg = reward_cfg or RewardConfig()
        self._last_info = {}
        self._last_lap_count = 0

    def step(self, action):
        obs, _, terminated, truncated, info = _compat_step(self.env, action)
        self._last_info = info

        lap_count = info.get("lap_count", 0)
        just_completed_lap = lap_count > self._last_lap_count
        self._last_lap_count = lap_count

        reward, crashed = compute_reward(info, self.reward_cfg, just_completed_lap)
        return obs, reward, terminated or crashed, truncated, info

    def reset(self, **kwargs):
        self._last_lap_count = 0
        return _compat_reset(self.env, **kwargs)

    def reward(self, reward):  # required by RewardWrapper's abstract interface; unused
        return reward


class ResizeObservation(ObservationWrapper):
    """Resizes raw camera frames to a consistent size before anything downstream sees them."""

    def __init__(self, env, size=(80, 160)):
        super().__init__(env)
        self.size = size  # (H, W)
        self.observation_space = spaces.Box(low=0, high=255, shape=(*size, 3), dtype=np.uint8)

    def observation(self, obs):
        return cv2.resize(np.asarray(obs), (self.size[1], self.size[0]))


class VAELatentObservationWrapper(ObservationWrapper):
    """Replaces image observations with the VAE's latent mean vector -- this is the
    observation SAC actually trains on. Loads a checkpoint produced by vae/train_vae.py."""

    def __init__(self, env, vae_checkpoint: str, device: str = "cpu"):
        super().__init__(env)
        print(f"[donkey_env] loading VAE checkpoint from {vae_checkpoint} ...", flush=True)
        ckpt = torch.load(vae_checkpoint, map_location=device)
        self.device = torch.device(device)
        self.vae = VAE(latent_dim=ckpt["latent_dim"]).to(self.device)
        self.vae.load_state_dict(ckpt["state_dict"])
        self.vae.eval()
        print("[donkey_env] VAE loaded", flush=True)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(ckpt["latent_dim"],), dtype=np.float32
        )

    def observation(self, obs):
        arr = np.asarray(obs, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1xCxHxW
        tensor = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode_mean(tensor)
        return latent.squeeze(0).cpu().numpy()


def make_raw_donkey_env(
    sim_path: str,
    host: str = "127.0.0.1",
    port: int = 9091,
    env_id: str = "donkey-generated-track-v0",
    image_size=(80, 160),
    max_episode_steps: int = 2000,
    reward_cfg: RewardConfig | None = None,
    frame_skip: int = 1,
) -> Env:
    """Raw env with resized image observations and our reward shaping -- what you'd use
    for frame collection or the PPO+CNN baseline."""
    import gym_donkeycar  # noqa: F401  (registers the donkey-* env ids on import)
    import gymnasium as gym

    # Tell Unity to render/send frames at our target size directly, instead of sending
    # 120x160 (the sim's own default) and throwing pixels away in ResizeObservation --
    # smaller frames over the socket is a real wall-clock speedup, not just a sample-
    # efficiency one.
    #
    # The sim's own game-over threshold is derived from reward_cfg.max_cte so it's the same
    # number you tune in configs/sac_vae.yaml, not a second hardcoded one living here.
    sim_max_cte = reward_cfg.max_cte if reward_cfg is not None else 8.0
    conf = _make_sim_conf(sim_path, host, port, frame_skip=frame_skip, cam_resolution=(*image_size, 3),
                           max_cte=sim_max_cte)
    print(f"[donkey_env] launching simulator at {sim_path} and connecting on {host}:{port} ...", flush=True)
    env = gym.make(env_id, conf=conf)
    print("[donkey_env] simulator connected, env created", flush=True)
    env = ResizeObservation(env, size=image_size)
    env = RewardShapingWrapper(env, reward_cfg)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_vae_env(
    sim_path: str,
    vae_checkpoint: str,
    host: str = "127.0.0.1",
    port: int = 9091,
    env_id: str = "donkey-generated-track-v0",
    image_size=(80, 160),
    max_episode_steps: int = 2000,
    reward_cfg: RewardConfig | None = None,
    device: str = "cpu",
    frame_skip: int = 1,
) -> Env:
    """The env SAC trains on: raw env -> resize -> reward shaping -> VAE latent observation."""
    env = make_raw_donkey_env(
        sim_path, host, port, env_id, image_size, max_episode_steps, reward_cfg, frame_skip
    )
    return VAELatentObservationWrapper(env, vae_checkpoint, device)


def _smoke_test(sim_path: str, host: str, port: int, env_id: str, steps: int):
    env = make_raw_donkey_env(sim_path, host, port, env_id)
    obs, info = _compat_reset(env)
    print(f"observation_space={env.observation_space}  action_space={env.action_space}")
    ep_reward = 0.0
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = _compat_step(env, action)
        ep_reward += reward
        if i % 50 == 0:
            print(f"step={i} reward={reward:.3f} cte={info.get('cte')} speed={info.get('speed')}")
        if terminated or truncated:
            print(f"episode ended at step {i}, total reward={ep_reward:.3f}")
            obs, info = _compat_reset(env)
            ep_reward = 0.0
    env.close()
    print("smoke test complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    _smoke_test(args.sim_path, args.host, args.port, args.env_id, args.steps)