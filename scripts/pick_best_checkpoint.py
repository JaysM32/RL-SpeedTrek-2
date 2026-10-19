"""Retroactively figures out which already-saved periodic checkpoint from a training run
performed best, for a run that finished (or is about to) BEFORE utils.callbacks.BestModelCallback
existed to track that automatically. Once you start a NEW training invocation with the current
agents/train_ppo_baseline.py / agents/train_sac.py, BestModelCallback does this for you going
forward and you won't need this script for that run.

Why this is needed at all: CheckpointCallback already saves a snapshot every --save-every-steps
steps throughout a run (checkpoints/ppo_waveshare/ppo_donkey_<step>_steps.zip, or the equivalent
sac_waveshare ones) -- but purely on a step schedule, not by how well the policy was driving at
that point. The final checkpoint a run ends on isn't guaranteed to be the best one (RL training
can get worse late in a run), and `--resume auto` always picks the highest-step-count file, which
is just the newest, not the best. This script loads each saved checkpoint, runs it for a few
deterministic episodes against the real env, and ranks them by mean episode reward so you can
pick the actual best one to carry into the next phase with an explicit `--resume <path>`.

Usage (PPO, evaluating a "safe"-mode run even if configs/sac_vae.yaml has since been changed
to "fast" in preparation for the next phase -- pass --mode to override):
    python -m scripts.pick_best_checkpoint --sim-path /path/to/donkey_sim \
        --algo ppo --config configs/sac_vae.yaml --mode safe \
        --checkpoint-glob "checkpoints/ppo_waveshare/ppo_donkey_*_steps.zip" \
        --final-checkpoint checkpoints/ppo_donkey.zip --episodes 5
        
        python -m scripts.pick_best_checkpoint --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" \
    --algo ppo --config configs/sac_vae.yaml --mode safe \
    --checkpoint-glob "checkpoints/ppo_waveshare/ppo_donkey_*_steps.zip" \
    --final-checkpoint checkpoints/ppo_donkey.zip --episodes 5 --max-episode-steps 500

Usage (SAC, needs --vae-checkpoint since make_vae_env requires one):
    python -m scripts.pick_best_checkpoint --sim-path /path/to/donkey_sim \
        --algo sac --vae-checkpoint checkpoints/vae.pt --config configs/sac_vae.yaml --mode safe \
        --checkpoint-glob "checkpoints/sac_waveshare/sac_donkey_*_steps.zip" \
        --final-checkpoint checkpoints/sac_donkey.zip --episodes 5

The ranked table is printed to the console AND the single best path is printed last on its own,
clearly labeled, so you can copy it straight into --resume <path> for the next training phase.
"""

import argparse
import glob
import re
import time

import numpy as np
import yaml

from utils.reward import RewardConfig


def _steps_from_path(path: str):
    match = re.search(r"_(\d+)_steps\.zip$", path)
    return int(match.group(1)) if match else None


def evaluate_checkpoint(model_cls, path: str, env, episodes: int, deterministic: bool = True) -> dict:
    model = model_cls.load(path, device="cpu")
    ep_rewards, ep_laps, crashes = [], [], 0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        ep_rewards.append(total_reward)
        ep_laps.append(info.get("lap_count_reliable", 0))
        if info.get("hit", "none") != "none":
            crashes += 1

    return {
        "path": path,
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_laps": float(np.mean(ep_laps)),
        "crash_rate": crashes / episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--config", default="configs/sac_vae.yaml")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--vae-checkpoint", default=None, help="required when --algo sac")
    parser.add_argument("--checkpoint-glob", required=True,
                         help='e.g. "checkpoints/ppo_waveshare/ppo_donkey_*_steps.zip"')
    parser.add_argument("--final-checkpoint", default=None,
                         help="also evaluate this one path (e.g. checkpoints/ppo_donkey.zip) "
                              "alongside everything --checkpoint-glob matches")
    parser.add_argument("--mode", default=None,
                         help="override reward.mode from --config -- use whatever mode the run "
                              "you're evaluating was ACTUALLY trained under, not whatever the "
                              "yaml currently says if you've since edited it for the next phase")
    parser.add_argument("--episodes", type=int, default=5,
                         help="deterministic evaluation episodes per checkpoint")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                         help="step budget per evaluation episode -- defaults to whatever "
                              "--config's env.max_episode_steps is (2000 in sac_vae.yaml, sized "
                              "for training). That's almost always more than you need just to "
                              "rank checkpoints against each other, and a competent/non-crashing "
                              "policy will burn the FULL budget on every single episode at real "
                              "simulator speed -- this is usually why evaluation takes much "
                              "longer than expected. A few hundred steps is normally plenty to "
                              "tell checkpoints apart; lower this for a much faster first pass.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.algo == "sac" and not args.vae_checkpoint:
        parser.error("--vae-checkpoint is required for --algo sac")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    reward_cfg_kwargs = dict(cfg["reward"])
    if args.mode:
        reward_cfg_kwargs["mode"] = args.mode
    reward_cfg = RewardConfig(**reward_cfg_kwargs)

    max_episode_steps = args.max_episode_steps or cfg["env"]["max_episode_steps"]

    print(f"evaluating under reward.mode={reward_cfg.mode!r} "
          f"({args.episodes} deterministic episode(s) per checkpoint, "
          f"max {max_episode_steps} step(s) each)\n")

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        from envs.donkey_env import make_raw_donkey_env
        model_cls = PPO
        env = make_raw_donkey_env(
            sim_path=args.sim_path,
            host=cfg["env"]["host"], port=cfg["env"]["port"], env_id=cfg["env"]["env_id"],
            image_size=tuple(cfg["env"]["image_size"][:2]),
            max_episode_steps=max_episode_steps,
            reward_cfg=reward_cfg, frame_skip=cfg["env"].get("frame_skip", 1),
        )
    else:
        from stable_baselines3 import SAC
        from envs.donkey_env import make_vae_env
        model_cls = SAC
        env = make_vae_env(
            sim_path=args.sim_path, vae_checkpoint=args.vae_checkpoint,
            host=cfg["env"]["host"], port=cfg["env"]["port"], env_id=cfg["env"]["env_id"],
            image_size=tuple(cfg["env"]["image_size"][:2]),
            max_episode_steps=max_episode_steps,
            reward_cfg=reward_cfg, device=args.device, frame_skip=cfg["env"].get("frame_skip", 1),
        )

    paths = sorted(glob.glob(args.checkpoint_glob), key=lambda p: (_steps_from_path(p) is None, _steps_from_path(p)))
    if args.final_checkpoint:
        paths.append(args.final_checkpoint)
    if not paths:
        env.close()
        raise SystemExit(f"no checkpoints matched --checkpoint-glob {args.checkpoint_glob!r} "
                          f"(and no --final-checkpoint given)")

    print(f"found {len(paths)} checkpoint(s) to evaluate: {', '.join(paths)}\n")

    # Printed per-checkpoint, not just once at the end -- each one can take a real while (see
    # --max-episode-steps' help text above), and with no output in between it's indistinguishable
    # from having hung. This also gives you a per-checkpoint time so you can estimate how long
    # the remaining ones will take, and a reason to Ctrl+C early if the first few are conclusive.
    results = []
    for i, p in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] evaluating {p} ...", flush=True)
        t0 = time.time()
        result = evaluate_checkpoint(model_cls, p, env, args.episodes)
        elapsed = time.time() - t0
        results.append(result)
        print(f"[{i}/{len(paths)}] done in {elapsed:.1f}s -- mean_reward={result['mean_reward']:.2f} "
              f"crash_rate={result['crash_rate']:.2f} mean_laps={result['mean_laps']:.2f}\n", flush=True)
    env.close()

    results.sort(key=lambda r: r["mean_reward"], reverse=True)

    header = f"{'checkpoint':60s} {'mean_reward':>12s} {'std':>8s} {'mean_laps':>10s} {'crash_rate':>10s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['path']:60s} {r['mean_reward']:12.2f} {r['std_reward']:8.2f} "
              f"{r['mean_laps']:10.2f} {r['crash_rate']:10.2f}")

    best = results[0]
    print(f"\nbest checkpoint by mean episode reward: {best['path']}")
    print(f"use this with --resume {best['path']} when you start the next training phase.")


if __name__ == "__main__":
    main()