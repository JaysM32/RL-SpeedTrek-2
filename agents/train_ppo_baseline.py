"""PPO + CNN baseline trained directly on raw camera frames (no VAE).

Useful as a comparison point against the VAE+SAC approach -- run both and compare sample
efficiency (reward vs. timesteps) and, if you get as far as real hardware, inference cost.

Usage:
    python -m agents.train_ppo_baseline --sim-path /path/to/donkey_sim --timesteps 300000
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.donkey_env import make_raw_donkey_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--out", default="checkpoints/ppo_donkey.zip")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = make_raw_donkey_env(args.sim_path, args.host, args.port, args.env_id)
    env = Monitor(env)

    # CnnPolicy expects channel-first images by default via SB3's VecTransposeImage when
    # using a VecEnv; for a single non-vectorized env SB3 handles the transpose internally.
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="runs/ppo_baseline",
        verbose=1,
        device=args.device,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"saved final model to {args.out}")
    env.close()


if __name__ == "__main__":
    main()
