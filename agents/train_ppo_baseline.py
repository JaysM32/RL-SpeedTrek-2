import argparse
import glob
import os
import re

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.donkey_env import make_raw_donkey_env
from utils.callbacks import BestModelCallback, EpisodeStatsCallback, make_checkpoint_callback
from utils.reward import RewardConfig


def _find_latest_checkpoint(save_dir: str, name_prefix: str) -> str | None:
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        default=None,
    )
    parser.add_argument(
        "--best-window", type=int, default=20,
    )
    parser.add_argument(
        "--best-min-episodes", type=int, default=5,
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)


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
        model.tensorboard_log = ppo_cfg["tensorboard_log"]  
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

    best_model_path = os.path.join(checkpoint_dir, f"best_model_{reward_cfg.mode}.zip")
    best_cb = BestModelCallback(
        save_path=best_model_path,
        window_size=args.best_window,
        min_episodes=args.best_min_episodes,
        verbose=1,
    )

    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        callback=[checkpoint_cb, stats_cb, best_cb],
        reset_num_timesteps=(resume_path is None),
    )

    model.save(ppo_cfg["checkpoint_out"])
    print(f"saved final model to {ppo_cfg['checkpoint_out']}")
    if os.path.exists(best_model_path):
        print(f"best model seen during THIS run (by rolling mean episode reward) saved to {best_model_path}")
    else:
        print(f"no 'best' checkpoint was saved to {best_model_path}")
    env.close()


if __name__ == "__main__":
    main()