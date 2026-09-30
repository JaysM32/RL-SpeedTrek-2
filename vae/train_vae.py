"""Train the VAE on collected frames.

Usage:
    python -m vae.train_vae --data data/frames --out checkpoints/vae.pt --latent-dim 32

Writes checkpoints/vae_reconstructions.png at the end so you can eyeball reconstruction
quality -- this is the single most useful debugging signal for this stage. If reconstructions
are unrecognizable noise, check that your frames actually contain varied, in-focus track
views before assuming the model architecture is the problem.
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vae.model import VAE, vae_loss


class FrameDataset(Dataset):
    def __init__(self, frame_dir: str, image_size=(80, 160)):
        self.paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"No .png frames found in {frame_dir}")
        self.image_size = image_size  # (H, W)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]))  # PIL wants (W, H)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="checkpoints/vae.pt")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kld-weight", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FrameDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = total_recon = total_kld = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar, args.kld_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()

        n_batches = len(loader)
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"loss={total_loss / n_batches:.4f}  "
            f"recon={total_recon / n_batches:.4f}  "
            f"kld={total_kld / n_batches:.4f}"
        )

    torch.save({"state_dict": model.state_dict(), "latent_dim": args.latent_dim}, args.out)
    print(f"saved VAE checkpoint to {args.out}")

    _save_reconstruction_grid(model, dataset, device, args.out)


def _save_reconstruction_grid(model, dataset, device, checkpoint_path, n=8):
    import matplotlib.pyplot as plt

    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    originals = torch.stack([dataset[i] for i in idxs]).to(device)
    with torch.no_grad():
        recon, _, _ = model(originals)

    fig, axes = plt.subplots(2, len(idxs), figsize=(2 * len(idxs), 4))
    for i in range(len(idxs)):
        axes[0, i].imshow(np.transpose(originals[i].cpu().numpy(), (1, 2, 0)))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.transpose(recon[i].cpu().numpy(), (1, 2, 0)))
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("recon")

    out_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "vae_reconstructions.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"saved reconstruction grid to {out_path}")


if __name__ == "__main__":
    main()
