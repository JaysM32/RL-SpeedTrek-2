"""Drive around the simulator collecting camera frames for VAE training.

Usage:
    python -m vae.collect_frames --sim-path /path/to/donkey_sim --out data/frames --n-frames 5000

    # to drive with a partially-trained SAC policy instead (usually covers even more of the
    # track once you have a checkpoint that's learned something):
    python -m vae.collect_frames --sim-path /path/to/donkey_sim --out data/frames --n-frames 5000 \
        --policy checkpoints/sac_intermediate/sac_donkey_40000_steps.zip

By default this drives with a *smoothed* random walk (small random deltas on steering each
step, constant moderate throttle) rather than fully independent random actions every step.
Fully independent random actions (env.action_space.sample() every step) steer hard-left then
hard-right every frame, which crashes almost immediately in practice -- if you used an earlier
version of this script, that's almost certainly why your frames only cover the first part of
the track. The smoothed driver survives much longer and sees much more of the track without
needing any trained policy at all.
"""

import argparse
import os

import numpy as np
from PIL import Image

from envs.donkey_env import make_raw_donkey_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--env-id", default="donkey-generated-track-v0")
    parser.add_argument("--out", default="data/frames")
    parser.add_argument("--n-frames", type=int, default=5000)
    parser.add_argument("--throttle", type=float, default=0.35,
                         help="constant throttle used by the smoothed random driver")
    parser.add_argument("--steer-step", type=float, default=0.15,
                         help="max change in steering per step for the smoothed random driver")
    parser.add_argument("--policy", default=None,
                         help="optional path to a SAC checkpoint .zip to drive with instead of "
                              "the smoothed random walk -- needs a VAE checkpoint too, since SAC "
                              "was trained on latent observations, not raw frames")
    parser.add_argument("--vae-checkpoint", default="checkpoints/vae.pt",
                         help="only used together with --policy")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Always build the *raw* env, even when driving with --policy -- that's the only way to
    # get the actual raw frame back from env.step() to save to disk. gym-donkeycar's `info`
    # dict does NOT carry the raw image (only pos/cte/speed/hit/gyro), so if we wrapped the
    # env in VAELatentObservationWrapper like training does, obs would already be a 32-d
    # latent vector with no way to recover the frame that produced it. Instead, when a
    # --policy is given, we encode each raw frame through the VAE ourselves, inline, just to
    # get the latent to feed model.predict() -- and still save the untouched raw frame.
    env = make_raw_donkey_env(args.sim_path, args.host, args.port, args.env_id)

    model = None
    vae = None
    device = None
    if args.policy:
        import torch
        from stable_baselines3 import SAC
        from vae.model import VAE

        device = torch.device("cpu")
        ckpt = torch.load(args.vae_checkpoint, map_location=device)
        vae = VAE(latent_dim=ckpt["latent_dim"]).to(device)
        vae.load_state_dict(ckpt["state_dict"])
        vae.eval()

        model = SAC.load(args.policy, device="cpu")
        print(f"[collect_frames] driving with policy from {args.policy}")
    else:
        steer = 0.0
        print("[collect_frames] driving with smoothed random walk (no policy)")

    def encode(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        tensor = torch.from_numpy(arr).to(device)
        with torch.no_grad():
            latent = vae.encode_mean(tensor)
        return latent.squeeze(0).cpu().numpy()

    obs, _ = env.reset()
    saved = 0
    while saved < args.n_frames:
        if model is not None:
            latent = encode(obs)
            action, _ = model.predict(latent, deterministic=False)
        else:
            # random walk on steering instead of resampling independently every step --
            # this is what keeps the car roughly on the road instead of snapping
            # hard-left/hard-right every frame and crashing in the first second.
            steer = float(np.clip(steer + np.random.uniform(-args.steer_step, args.steer_step), -1.0, 1.0))
            action = np.array([steer, args.throttle], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)  # obs is the raw frame here

        frame = Image.fromarray(np.array(obs))
        frame.save(os.path.join(args.out, f"frame_{saved:06d}.png"))
        saved += 1

        if saved % 500 == 0:
            print(f"collected {saved}/{args.n_frames} frames")

        if terminated or truncated:
            obs, _ = env.reset()
            if model is None:
                steer = 0.0

    env.close()
    print(f"done. {saved} frames saved to {args.out}")


if __name__ == "__main__":
    main()