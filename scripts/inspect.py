"""Step through the env and print/save the actual state, action, and reward at every step --
useful for sanity-checking that the reward function and VAE latent state look reasonable,
and for screenshots/plots in your assignment writeup.

Usage:
    # random actions, prints live and saves a CSV
    python -m scripts.inspect_transitions --sim-path /path/to/donkey_sim \
        --vae-checkpoint checkpoints/vae.pt --steps 300

    # drive with a trained policy instead of random actions
    python -m scripts.inspect_transitions --sim-path /path/to/donkey_sim \
        --vae-checkpoint checkpoints/vae.pt --policy checkpoints/sac_donkey.zip --steps 300

"state" here means the VAE latent vector (32 numbers by default) -- that's what SAC actually
sees and trains on, not the raw image. The 32 numbers on their own aren't human-readable, so
this also prints/saves cte and speed from `info` alongside them so you can sanity-check the
state against something you can interpret (e.g. "state changed a lot right as cte spiked").
"""

import argparse
import csv

from envs.donkey_env import make_vae_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env-id", default="donkey-circuit-launch-track-v0")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--policy", default=None,
                         help="optional SAC checkpoint .zip -- omit to use random actions")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", default="logs/transitions.csv")
    parser.add_argument("--print-every", type=int, default=1,
                         help="print a line to the console every N steps (still logs every step to CSV)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = make_vae_env(
        sim_path=args.sim_path,
        vae_checkpoint=args.vae_checkpoint,
        host=args.host,
        port=args.port,
        env_id=args.env_id,
        device=args.device,
    )

    model = None
    if args.policy:
        from stable_baselines3 import SAC
        model = SAC.load(args.policy, device=args.device)
        print(f"[inspect] driving with policy from {args.policy}")
    else:
        print("[inspect] driving with random actions (pass --policy to use a trained model)")

    latent_dim = env.observation_space.shape[0]
    header = (
        ["step", "steer", "throttle", "reward", "terminated", "truncated", "cte", "speed", "hit"]
        + [f"state_{i}" for i in range(latent_dim)]
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    obs, info = env.reset()
    rows = []
    for step in range(args.steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        steer, throttle = float(action[0]), float(action[1])
        cte = info.get("cte")
        speed = info.get("speed")
        hit = info.get("hit")

        if step % args.print_every == 0:
            print(
                f"step={step:4d}  steer={steer:+.3f} throttle={throttle:.3f}  "
                f"reward={reward:+.3f}  cte={cte}  speed={speed}  hit={hit}"
            )

        rows.append(
            [step, steer, throttle, reward, terminated, truncated, cte, speed, hit]
            + list(obs)
        )

        if terminated or truncated:
            print(f"  -- episode ended at step {step}, resetting --")
            obs, info = env.reset()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    env.close()
    print(f"\nsaved {len(rows)} steps to {args.out}")
    print("columns: step, steer, throttle, reward, terminated, truncated, cte, speed, hit, "
          f"state_0..state_{latent_dim - 1} (the VAE latent vector)")


if __name__ == "__main__":
    main()