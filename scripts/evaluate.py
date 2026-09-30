"""Run evaluation episodes for a trained SAC (VAE-latent) policy and report summary stats.

Usage:
    python -m scripts.evaluate --sim-path /path/to/donkey_sim --model checkpoints/sac_donkey.zip \
        --vae-checkpoint checkpoints/vae.pt --episodes 10
"""

import argparse

import numpy as np
from stable_baselines3 import SAC

from envs.donkey_env import make_vae_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = make_vae_env(
        sim_path=args.sim_path,
        vae_checkpoint=args.vae_checkpoint,
        env_id=args.env_id,
        device=args.device,
    )
    model = SAC.load(args.model, device=args.device)

    rewards, lengths, crashes = [], [], 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
        if info.get("hit", "none") != "none":
            crashes += 1
        print(f"episode {ep + 1}: reward={ep_reward:.2f} length={ep_len}")

    env.close()
    print("\n--- summary ---")
    print(f"mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"mean episode length: {np.mean(lengths):.1f}")
    print(f"crash rate: {crashes}/{args.episodes}")


if __name__ == "__main__":
    main()
