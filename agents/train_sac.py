"""Train SAC on top of VAE latent observations. This is the main training script.

Usage:
    python -m agents.train_sac --sim-path /path/to/donkey_sim --vae-checkpoint checkpoints/vae.pt \
        --config configs/sac_vae.yaml

    # to resume the most recent checkpoint instead of starting over:
    python -m agents.train_sac --sim-path /path/to/donkey_sim --vae-checkpoint checkpoints/vae.pt \
        --config configs/sac_vae.yaml --resume auto

Requires a VAE checkpoint from vae/train_vae.py -- train that first.

IMPORTANT: by default this always starts a brand-new model from scratch, even if
checkpoints already exist in checkpoints/sac_intermediate/ -- it does not auto-resume.
Pass --resume auto (pick up the latest checkpoint) or --resume <path-to-checkpoint.zip>
(a specific one) if you want to continue an interrupted run instead of losing that progress.
"""

import argparse
import glob
import os
import re

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from envs.donkey_env import make_vae_env
from utils.callbacks import EpisodeStatsCallback, make_checkpoint_callback
from utils.reward import RewardConfig


def _find_latest_checkpoint(save_dir: str, name_prefix: str) -> str | None:
    """Finds the highest-timestep checkpoint written by CheckpointCallback, e.g.
    checkpoints/sac_intermediate/sac_donkey_40000_steps.zip -> picks 40000 over 30000."""
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
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--config", default="configs/sac_vae.yaml")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--resume",
        default=None,
        help="'auto' to resume the latest checkpoint in checkpoints/sac_intermediate/, "
        "or a specific checkpoint .zip path. Omit to always start fresh (the old default behavior).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    reward_cfg = RewardConfig(**cfg["reward"])

    print("[train_sac] building environment (this launches the simulator) ...", flush=True)
    env = make_vae_env(
        sim_path=args.sim_path,
        vae_checkpoint=args.vae_checkpoint,
        host=cfg["env"]["host"],
        port=cfg["env"]["port"],
        env_id=cfg["env"]["env_id"],
        image_size=tuple(cfg["env"]["image_size"][:2]),
        max_episode_steps=cfg["env"]["max_episode_steps"],
        reward_cfg=reward_cfg,
        device=args.device,
        frame_skip=cfg["env"].get("frame_skip", 1),
    )
    env = Monitor(env)
    print("[train_sac] environment ready", flush=True)

    sac_cfg = cfg["sac"]
    checkpoint_dir = "checkpoints/sac_waveshare"

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = _find_latest_checkpoint(checkpoint_dir, "sac_donkey")
        if resume_path is None:
            print(f"--resume auto: no checkpoints found in {checkpoint_dir}, starting fresh instead.")

    if resume_path:
        print(f"[train_sac] resuming from {resume_path} -- loading model ...", flush=True)
        model = SAC.load(resume_path, env=env, device=args.device)
        print("[train_sac] model loaded, resuming training", flush=True)
        model.tensorboard_log = sac_cfg["tensorboard_log"]  # keep logging to the same run dir
    else:
        print("starting a new model from scratch (no --resume given)")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            batch_size=sac_cfg["batch_size"],
            gamma=sac_cfg["gamma"],
            tau=sac_cfg["tau"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            learning_starts=sac_cfg["learning_starts"],
            policy_kwargs=sac_cfg["policy_kwargs"],
            tensorboard_log=sac_cfg["tensorboard_log"],
            verbose=1,
            device=args.device,
        )

    checkpoint_cb = make_checkpoint_callback(
        save_freq=sac_cfg["save_every_steps"],
        save_path=checkpoint_dir,
        name_prefix="sac_donkey",
    )
    stats_cb = EpisodeStatsCallback()

    # reset_num_timesteps=False is what makes a resumed run keep counting from where the
    # loaded checkpoint left off, instead of resetting the step counter (and learning_starts
    # warm-up) back to zero.
    model.learn(
        total_timesteps=sac_cfg["total_timesteps"],
        callback=[checkpoint_cb, stats_cb],
        reset_num_timesteps=(resume_path is None),
    )

    model.save(sac_cfg["checkpoint_out"])
    print(f"saved final model to {sac_cfg['checkpoint_out']}")
    env.close()


if __name__ == "__main__":
    main()