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
    parser.add_argument("--throttle", type=float, default=0.35)
    parser.add_argument("--steer-step", type=float, default=0.15)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--vae-checkpoint", default="checkpoints/vae.pt")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    
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
            steer = float(np.clip(steer + np.random.uniform(-args.steer_step, args.steer_step), -1.0, 1.0))
            action = np.array([steer, args.throttle], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)  

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