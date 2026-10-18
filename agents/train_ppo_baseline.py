"""Train PPO + CNN directly on raw camera frames (no VAE) -- the comparison baseline
against the VAE+SAC approach in agents/train_sac.py.

Trained the *same way* as train_sac.py: same configs/sac_vae.yaml env:/reward: sections
(same track, same frame_skip, same max_cte, same safe/fast reward curriculum), the same
EpisodeStatsCallback lap-time logging, the same CheckpointCallback periodic saving, and
the same --resume auto mechanics. The one deliberate difference is the observation space:
this trains CnnPolicy directly on resized camera frames, SAC trains on VAE latent vectors
-- that difference is the whole point of this baseline (sample efficiency / inference cost
with vs. without the VAE), so don't "fix" it away.

Usage:
    python -m agents.train_ppo_baseline --sim-path /path/to/donkey_sim --config configs/sac_vae.yaml

    # to resume the most recent checkpoint instead of starting over:
    python -m agents.train_ppo_baseline --sim-path /path/to/donkey_sim \
        --config configs/sac_vae.yaml --resume auto

IMPORTANT: by default this always starts a brand-new model from scratch, even if
checkpoints already exist in checkpoints/ppo_waveshare/ -- it does not auto-resume.
Pass --resume auto (pick up the latest checkpoint) or --resume <path-to-checkpoint.zip>
(a specific one) if you want to continue an interrupted run instead of losing that progress.
"""

import argparse
import glob
import os
import re

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.donkey_env import make_raw_donkey_env
from utils.callbacks import EpisodeStatsCallback, make_checkpoint_callback
from utils.reward import RewardConfig


def _find_latest_checkpoint(save_dir: str, name_prefix: str) -> str | None:
    """Finds the highest-timestep checkpoint written by CheckpointCallback, e.g.
    checkpoints/ppo_waveshare/ppo_donkey_40000_steps.zip -> picks 40000 over 30000."""
    pattern = os.path.join(save_dir, f"{name_prefix}_*_steps.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def step_count(path: str) -> int:
        match = re.search(r"_(\d+)_steps\.zip$", path)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--config", default="configs/sac_vae.yaml")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--resume",
        default=None,
        help="'auto' to resume the latest checkpoint in checkpoints/ppo_waveshare/, "
        "or a specific checkpoint .zip path. Omit to always start fresh (the old default behavior).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Same RewardConfig construction as train_sac.py, from the same cfg["reward"] section --
    # this is what fixes the baseline being silently frozen in "safe" mode on the dataclass's
    # bare defaults: it now sees whatever mode/max_cte/etc. the yaml currently has (right now:
    # mode="fast", max_cte=1.5), same as SAC does.
    reward_cfg = RewardConfig(**cfg["reward"])

    print("[train_ppo_baseline] building environment (this launches the simulator) ...", flush=True)
    env = make_raw_donkey_env(
        sim_path=args.sim_path,
        host=cfg["env"]["host"],
        port=cfg["env"]["port"],
        env_id=cfg["env"]["env_id"],
        image_size=tuple(cfg["env"]["image_size"][:2]),
        max_episode_steps=cfg["env"]["max_episode_steps"],
        reward_cfg=reward_cfg,
        frame_skip=cfg["env"].get("frame_skip", 1),
    )
    env = Monitor(env)
    print("[train_ppo_baseline] environment ready", flush=True)

    ppo_cfg = cfg["ppo"]
    checkpoint_dir = "checkpoints/ppo_waveshare"

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = _find_latest_checkpoint(checkpoint_dir, "ppo_donkey")
        if resume_path is None:
            print(f"--resume auto: no checkpoints found in {checkpoint_dir}, starting fresh instead.")

    if resume_path:
        print(f"[train_ppo_baseline] resuming from {resume_path} -- loading model ...", flush=True)
        model = PPO.load(resume_path, env=env, device=args.device)
        print("[train_ppo_baseline] model loaded, resuming training", flush=True)
        model.tensorboard_log = ppo_cfg["tensorboard_log"]  # keep logging to the same run dir
    else:
        print("starting a new model from scratch (no --resume given)")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            gamma=ppo_cfg["gamma"],
            tensorboard_log=ppo_cfg["tensorboard_log"],
            verbose=1,
            device=args.device,
        )

    checkpoint_cb = make_checkpoint_callback(
        save_freq=ppo_cfg["save_every_steps"],
        save_path=checkpoint_dir,
        name_prefix="ppo_donkey",
    )
    stats_cb = EpisodeStatsCallback()

    # reset_num_timesteps=False is what makes a resumed run keep counting from where the
    # loaded checkpoint left off, instead of resetting the step counter (and PPO's own
    # rollout-collection bookkeeping) back to zero -- identical to train_sac.py.
    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        callback=[checkpoint_cb, stats_cb],
        reset_num_timesteps=(resume_path is None),
    )

    model.save(ppo_cfg["checkpoint_out"])
    print(f"saved final model to {ppo_cfg['checkpoint_out']}")
    env.close()


if __name__ == "__main__":
    main()