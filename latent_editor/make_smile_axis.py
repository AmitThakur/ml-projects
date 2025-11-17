#!/usr/bin/env python3
# make_smile_axis.py (noise-aware, spatial-latent)

"""
python make_smile_axis.py \
  --pairs_dir ./pairs \
  --ckpt files/hollie-mengert.ckpt \
  --device cuda \
  --image_size 256 --center_crop \
  --norm_to minus1_1 --per_pair_unit \
  --save_axis smile_axis.pt \
  --plot3d smile_axis_3d.png \
  --plot_pair_arrows --max_pair_arrows 120 \
  --animate3d smile_axis_3d.mp4 \
  --animate_pair_arrows --max_animate_pair_arrows 100 \
  --animate_arrow_uniform --animate_arrow_scale 0.4 \
  --apply_in ./test_faces \
  --apply_out ./edited_faces \
  --alphas -0.1 0.0 0.1 0.2

"""
import sys

sys.path.append("pytorch-stable-diffusion/sd")

from model_converter import load_from_standard_weights
from encoder import VAE_Encoder
from decoder import VAE_Decoder

import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = torch.distributions.Normal(0, 1)


# -----------------------
# Utils: transforms / IO
# -----------------------
def build_transform(image_size: int, center_crop: bool, norm_to: str):
    t: List[transforms.Transform] = []
    if center_crop:
        t.append(transforms.CenterCrop(image_size))
    t.append(transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.ToTensor())
    tfm = transforms.Compose(t)

    def post_norm(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1] from ToTensor
        if norm_to == "minus1_1":
            return x * 2.0 - 1.0
        return x  # keep [0,1]
    return tfm, post_norm


def load_image(path: Path, tfm, post_norm, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = tfm(img)             # [C,H,W] in [0,1]
    x = post_norm(x)         # [C,H,W] in chosen range
    return x.unsqueeze(0).to(device)  # [1,C,H,W]


def unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


# -----------------------
# Noise-aware encoding
# -----------------------
def parse_noise_shape(s: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse a shape like '1,4,64,64' -> (1,4,64,64).
    Empty string means None (no noise expected by encoder).
    """
    s = s.strip()
    if not s:
        return None
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("noise_shape must be like '1,4,64,64'")
    return tuple(parts)  # type: ignore


def auto_noise_shape_from_image(x: torch.Tensor, latent_ch: int = 4, down_factor: int = 8):
    # x: [B,C,H,W] input batch to encoder
    H, W = x.shape[-2], x.shape[-1]
    return (x.shape[0], latent_ch, H // down_factor, W // down_factor)


@torch.no_grad()
def encode_latent(
    encoder: torch.nn.Module,
    x: torch.Tensor,                           # [B,C,H,W]
    device: torch.device,
    noise_shape: Optional[Tuple[int, int, int, int]] = None
) -> torch.Tensor:
    """
    Encodes images to latent. Supports two encoder signatures:
    - encoder(x) -> latent
    - encoder(x, noise) -> latent  (spatial VAEs)
    Returns latent tensor (could be spatial), shape e.g. [B,4,64,64] or [B,d].
    """
    if noise_shape is None or noise_shape == ():
        shape = auto_noise_shape_from_image(x.to(device))
    else:
        shape = list(noise_shape);
        shape[0] = x.size(0)
    noise = N.sample(torch.Size(shape)).to(device)
    out = encoder(x.to(device), noise)

    # Handle various output structures; prefer "mu"/mean if provided
    if isinstance(out, (tuple, list)):
        z = out[0]
    elif isinstance(out, dict):
        z = out.get("mu", out.get("mean", out.get("latent", None)))
        if z is None:
            raise RuntimeError("Encoder dict output missing 'mu'/'mean'/'latent'.")
    else:
        z = out
    return z


# -----------------------
# Axis computation (pairs)
# -----------------------
@torch.no_grad()
def compute_smile_axis_from_pairs(
    encoder: torch.nn.Module,
    pairs_csv: Path,
    img_dir: Path,
    image_size: int,
    center_crop: bool,
    norm_to: str,
    device: torch.device,
    noise_shape: Optional[Tuple[int, int, int, int]],
    batch: int,
    per_pair_unit: bool,
) -> Tuple[torch.Tensor, dict]:
    """
    Computes the global smile axis from neutral/smile pairs.
    Works with spatial latents; flattens to vectors, averages, normalizes.
    """
    df = pd.read_csv(pairs_csv)
    tfm, post_norm = build_transform(image_size, center_crop, norm_to)

    diffs = []
    pair_norms = []

    rows = df.to_dict("records")
    for i in range(0, len(rows), batch):
        chunk = rows[i:i+batch]

        # load images
        neu_list, smi_list = [], []
        for r in chunk:
            neu_list.append(load_image(img_dir / r["neutral_file"], tfm, post_norm, device)[0])
            smi_list.append(load_image(img_dir / r["smile_file"],   tfm, post_norm, device)[0])

        xb_neu = torch.stack(neu_list, dim=0)  # [B,C,H,W]
        xb_smi = torch.stack(smi_list, dim=0)

        # IMPORTANT: use the SAME noise for neutral/smile in this batch
        if noise_shape is None or noise_shape == ():
            shape = auto_noise_shape_from_image(xb_neu)  # (B,4,H//8,W//8)
        else:
            shape = list(noise_shape);
            shape[0] = xb_neu.size(0)
        noise = N.sample(torch.Size(shape)).to(device)
        z_neu = encoder(xb_neu, noise)
        z_smi = encoder(xb_smi, noise)

        # flatten spatial latents to vectors
        z_neu_flat = z_neu.view(z_neu.size(0), -1)  # [B,d]
        z_smi_flat = z_smi.view(z_smi.size(0), -1)

        v_batch = z_smi_flat - z_neu_flat           # [B,d]
        if per_pair_unit:
            v_batch = unit(v_batch)
        diffs.append(v_batch)
        pair_norms.append(v_batch.norm(dim=-1).detach().cpu())

    V = torch.cat(diffs, dim=0)                     # [N,d]
    axis = V.mean(dim=0)                            # [d]
    axis = axis / (axis.norm() + 1e-9)

    stats = {
        "num_pairs": len(df),
        "per_pair_unit": per_pair_unit,
        "pair_norm_mean": torch.cat(pair_norms).mean().item() if pair_norms else 0.0,
        "pair_norm_std":  torch.cat(pair_norms).std(unbiased=False).item() if pair_norms else 0.0,
        "axis_dim": axis.numel(),
    }
    return axis, stats


# -----------------------
# Editing (flattened tangent or linear)
# -----------------------
@torch.no_grad()
def tangent_edit_flat(z_flat: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Norm-preserving spherical move in flattened latent space.
    z_flat, v: [d]
    """
    zn = z_flat.norm()
    if zn < 1e-9:
        return z_flat + alpha * v
    z_hat = z_flat / zn
    v_perp = v - (z_hat @ v) * z_hat
    vn = v_perp.norm()
    if vn < 1e-9:
        return z_flat
    v_hat = v_perp / vn
    ca = torch.cos(torch.tensor(alpha, device=z_flat.device))
    sa = torch.sin(torch.tensor(alpha, device=z_flat.device))
    return zn * (ca * z_hat + sa * v_hat)


@torch.no_grad()
def apply_axis_to_folder(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    axis: torch.Tensor,              # [d] flattened axis
    in_dir: Path,
    out_dir: Path,
    alphas: List[float],
    image_size: int,
    center_crop: bool,
    norm_to: str,
    device: torch.device,
    noise_shape: Optional[Tuple[int, int, int, int]],
    tangent: bool = True,
    grid: bool = True,
    nrow: Optional[int] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    tfm, post_norm = build_transform(image_size, center_crop, norm_to)
    axis = axis.to(device)

    img_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not img_files:
        print(f"No images found in {in_dir}")
        return

    for p in img_files:
        x = load_image(p, tfm, post_norm, device)  # [1,C,H,W]

        # Encode once (with noise if required)
        z = encode_latent(encoder, x, device, noise_shape)  # shape [1,*,*,*] or [1,d]
        z_flat = z.view(-1)                                 # [d]

        outs = []
        for a in alphas:
            if tangent:
                z_edit_flat = tangent_edit_flat(z_flat, axis, a)
            else:
                z_edit_flat = z_flat + a * axis
            z_edit = z_edit_flat.view_as(z)                 # reshape back to latent shape
            x_out = decoder(z_edit)                         # [1,C,H,W]
            # map to [0,1] for saving if model is in [-1,1]
            if norm_to == "minus1_1":
                x_vis = (x_out.clamp(-1, 1) + 1) / 2.0
            else:
                x_vis = x_out.clamp(0, 1)
            outs.append(x_vis)

        if grid:
            g = make_grid(torch.cat(outs, dim=0), nrow=len(alphas) if nrow is None else nrow)
            save_image(g, out_dir / f"{p.stem}_smile_grid.png")
        else:
            for a, x_out in zip(alphas, outs):
                save_image(x_out, out_dir / f"{p.stem}_alpha{a:+.2f}.png")


# -----------------------
# PCA → 3D + Plot/Animate
# -----------------------
@torch.no_grad()
def pca_project_3d(X: torch.Tensor):
    """
    X: [N, d] (flattened latents)
    Returns X3: [N,3], P: [d,3], mu: [d]
    """
    mu = X.mean(dim=0, keepdim=True)    # [1,d]
    Xc = X - mu
    try:
        U, S, V = torch.pca_lowrank(Xc, q=3, center=False)
        P = V[:, :3]                    # [d,3]
    except Exception:
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        P = Vt.T[:, :3]
    X3 = Xc @ P                         # [N,3]
    return X3, P, mu[0]


@torch.no_grad()
def plot_axis_3d_from_pairs(
    encoder,
    pairs_csv: Path,
    img_dir: Path,
    axis: torch.Tensor,                 # [d] flattened
    image_size: int,
    center_crop: bool,
    norm_to: str,
    device: torch.device,
    noise_shape: Optional[Tuple[int, int, int, int]],
    out_png: Path,
    max_pairs: int = 300,
    plot_pair_arrows: bool = False,
    max_pair_arrows: int = 150,
):
    import numpy as np
    df = pd.read_csv(pairs_csv)
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=0).reset_index(drop=True)

    tfm, post_norm = build_transform(image_size, center_crop, norm_to)

    Z_neu, Z_smi = [], []
    for r in df.to_dict("records"):
        xn = load_image(img_dir / r["neutral_file"], tfm, post_norm, device)
        xs = load_image(img_dir / r["smile_file"],   tfm, post_norm, device)
        zn = encode_latent(encoder, xn, device, noise_shape)[0].view(-1)
        zs = encode_latent(encoder, xs, device, noise_shape)[0].view(-1)
        Z_neu.append(zn)
        Z_smi.append(zs)

    Z_neu = torch.stack(Z_neu, dim=0)     # [N,d]
    Z_smi = torch.stack(Z_smi, dim=0)     # [N,d]
    Z_all = torch.cat([Z_neu, Z_smi], dim=0)

    # PCA → 3D
    X3, P, _mu = pca_project_3d(Z_all)    # [2N,3]
    N = Z_neu.size(0)
    Xn3, Xs3 = X3[:N], X3[N:]

    # Global axis projected to 3D
    v = axis / (axis.norm() + 1e-9)
    v3 = (P.T @ v).cpu().numpy()
    v3 /= np.linalg.norm(v3) + 1e-9

    mu3   = X3.mean(dim=0).cpu().numpy()
    scale = float(X3.std(dim=0).mean().cpu().item() * 2.0)

    # ---- Plot ----
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter clouds
    s0 = ax.scatter(Xn3[:, 0].cpu(), Xn3[:, 1].cpu(), Xn3[:, 2].cpu(), s=8, label="neutral")
    s1 = ax.scatter(Xs3[:, 0].cpu(), Xs3[:, 1].cpu(), Xs3[:, 2].cpu(), s=8, marker="^", label="smile")

    # Global direction arrow (red)
    ax.quiver(mu3[0], mu3[1], mu3[2], v3[0], v3[1], v3[2],
              length=scale, arrow_length_ratio=0.15, linewidth=2, color='red')

    # Per-pair arrows (neutral → smile), lightly styled to avoid clutter
    if plot_pair_arrows and N > 0:
        # Subsample indices if there are too many
        idx = np.arange(N)
        if N > max_pair_arrows:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=max_pair_arrows, replace=False)
        # Draw each as a quiver starting at neutral point with direction to smile point
        Xn_np = Xn3.cpu().numpy()
        Xs_np = Xs3.cpu().numpy()
        for i in idx:
            d = Xs_np[i] - Xn_np[i]
            # Optionally normalize arrow length for visual consistency:
            # d = d / (np.linalg.norm(d) + 1e-9) * (scale * 0.35)
            ax.quiver(Xn_np[i,0], Xn_np[i,1], Xn_np[i,2],
                      d[0], d[1], d[2],
                      arrow_length_ratio=0.15, linewidth=1,
                      color='gray', alpha=0.35)

        # add a legend proxy for arrows
        from matplotlib.lines import Line2D
        proxy = Line2D([0],[0], color='gray', alpha=0.6, lw=2)
        leg = ax.legend([s0, s1, proxy], ["neutral", "smile", "pair arrows"], loc="upper left")
    else:
        ax.legend(loc="upper left")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Smile Axis in Latent Space (3D PCA Projection)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[OK] 3D plot with per-pair arrows saved to: {out_png}")




@torch.no_grad()
def animate_axis_3d_from_pairs(
    encoder,
    pairs_csv: Path,
    img_dir: Path,
    axis: torch.Tensor,                 # [d] flattened
    image_size: int,
    center_crop: bool,
    norm_to: str,
    device: torch.device,
    noise_shape: Optional[Tuple[int, int, int, int]],
    out_vid: Path,
    max_pairs: int = 300,
    seconds: float = 6.0,
    fps: int = 24,
    elev: float = 20.0,
    azim_start: float = 45.0,
    spin_degrees: float = 360.0,
    animate_pair_arrows: bool = False,
    max_animate_pair_arrows: int = 120,
    animate_arrow_uniform: bool = False,
    animate_arrow_scale: float = 0.35,
):
    import numpy as np
    df = pd.read_csv(pairs_csv)
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=0).reset_index(drop=True)

    tfm, post_norm = build_transform(image_size, center_crop, norm_to)

    Z_neu, Z_smi = [], []
    for r in df.to_dict("records"):
        xn = load_image(img_dir / r["neutral_file"], tfm, post_norm, device)
        xs = load_image(img_dir / r["smile_file"],   tfm, post_norm, device)
        zn = encode_latent(encoder, xn, device, noise_shape)[0].view(-1)
        zs = encode_latent(encoder, xs, device, noise_shape)[0].view(-1)
        Z_neu.append(zn)
        Z_smi.append(zs)

    Z_neu = torch.stack(Z_neu, dim=0)
    Z_smi = torch.stack(Z_smi, dim=0)
    Z_all = torch.cat([Z_neu, Z_smi], dim=0)

    # PCA → 3D
    X3, P, _mu = pca_project_3d(Z_all)
    N = Z_neu.size(0)
    Xn3, Xs3 = X3[:N], X3[N:]

    # Global axis projected to 3D
    v = axis / (axis.norm() + 1e-9)
    v3 = (P.T @ v).cpu().numpy()
    v3 /= np.linalg.norm(v3) + 1e-9

    mu3   = X3.mean(dim=0).cpu().numpy()
    scale = float(X3.std(dim=0).mean().cpu().item() * 2.0)

    # ---- Build figure once ----
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    scat_neu = ax.scatter(Xn3[:, 0].cpu(), Xn3[:, 1].cpu(), Xn3[:, 2].cpu(), s=8, label="neutral")
    scat_smi = ax.scatter(Xs3[:, 0].cpu(), Xs3[:, 1].cpu(), Xs3[:, 2].cpu(), s=8, marker="^", label="smile")

    # Global arrow (red)
    quiv_global = ax.quiver(mu3[0], mu3[1], mu3[2], v3[0], v3[1], v3[2],
                            length=scale, arrow_length_ratio=0.15, linewidth=2, color='red')

    # Optional per-pair arrows (neutral → smile) — drawn once, just rotate camera each frame
    pair_quivers = []
    if animate_pair_arrows and N > 0:
        Xn_np = Xn3.cpu().numpy()
        Xs_np = Xs3.cpu().numpy()
        idx = np.arange(N)
        if N > max_animate_pair_arrows:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, size=max_animate_pair_arrows, replace=False)
        for i in idx:
            start = Xn_np[i]
            d = Xs_np[i] - start
            if animate_arrow_uniform:
                n = np.linalg.norm(d) + 1e-9
                d = (d / n) * (scale * animate_arrow_scale)
            q = ax.quiver(start[0], start[1], start[2],
                          d[0], d[1], d[2],
                          arrow_length_ratio=0.15, linewidth=1,
                          color='gray', alpha=0.35)
            pair_quivers.append(q)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Smile Axis in Latent Space (3D PCA Projection)")
    ax.legend(loc="upper left")
    plt.tight_layout()

    frames = max(1, int(seconds * fps))
    def update(f):
        azim = azim_start + (spin_degrees * f / frames)
        ax.view_init(elev=elev, azim=azim)
        # Artists returned (global + scatters + pair arrows)
        return (scat_neu, scat_smi, quiv_global, *pair_quivers)

    ext = out_vid.suffix.lower()
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    if ext == ".mp4":
        try:
            Writer = animation.FFMpegWriter
            writ = Writer(fps=fps, metadata={"title": "Smile Axis 3D"}, bitrate=3000)
            ani.save(out_vid, writer=writ, dpi=220)
        except Exception as e:
            print(f"[warn] ffmpeg unavailable? Falling back to GIF. ({e})")
            out_vid = out_vid.with_suffix(".gif")
            ani.save(out_vid, writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(out_vid, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[OK] 3D animation saved to: {out_vid}")



# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Compute global smile axis from CelebA pairs (noise-aware spatial VAE) and apply/visualize it.")
    ap.add_argument("--pairs_dir", type=str, required=True, help="Directory containing pairs.csv and images.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to VAE checkpoint (e.g., files/hollie-mengert.ckpt).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--center_crop", action="store_true")
    ap.add_argument("--norm_to", type=str, default="minus1_1", choices=["minus1_1", "zero1"],
                    help="[-1,1] or [0,1] preprocessing. Must match VAE training.")
    ap.add_argument("--per_pair_unit", action="store_true",
                    help="Normalize each pair difference before averaging.")
    ap.add_argument("--batch", type=int, default=16)

    # Noise / latent specifics
    ap.add_argument("--noise_shape", type=str, default="",
                    help="Noise shape 'B,C,H,W'. Leave empty to auto-compute (uses down_factor=8, latent_ch=4).")

    # Save outputs
    ap.add_argument("--save_axis", type=str, default="smile_axis.pt")
    ap.add_argument("--meta_json", type=str, default="smile_axis_meta.json")

    # Apply to folder
    ap.add_argument("--apply_in", type=str, default="", help="Folder of images to edit (optional).")
    ap.add_argument("--apply_out", type=str, default="edited", help="Output folder for edited images.")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-0.6, -0.3, 0.0, 0.3, 0.6])
    ap.add_argument("--linear_move", action="store_true", help="Use linear z' = z + alpha v (default: tangent).")

    # Visualization
    ap.add_argument("--plot3d", type=str, default="", help="Filename to save 3D PCA plot (e.g., smile_axis_3d.png)")
    # --- New static-plot flags for per-pair arrows ---
    ap.add_argument("--plot_pair_arrows", action="store_true",
                    help="Draw neutral→smile arrows for each pair in the 3D PCA plot.")
    ap.add_argument("--max_pair_arrows", type=int, default=150,
                    help="Maximum number of per-pair arrows to draw (subsample if more).")


    ap.add_argument("--animate3d", type=str, default="", help="Filename to save rotating 3D PCA (e.g., smile_axis_3d.mp4 or .gif)")
    ap.add_argument("--anim_seconds", type=float, default=6.0)
    ap.add_argument("--anim_fps", type=int, default=24)
    ap.add_argument("--anim_elev", type=float, default=20.0)
    ap.add_argument("--anim_azim_start", type=float, default=45.0)
    ap.add_argument("--anim_spin_degrees", type=float, default=360.0)
    ap.add_argument("--animate_pair_arrows", action="store_true",
                    help="Draw neutral→smile arrows for each pair in the 3D animation.")
    ap.add_argument("--max_animate_pair_arrows", type=int, default=120,
                    help="Max per-pair arrows to draw in animation (subsample if more).")
    ap.add_argument("--animate_arrow_uniform", action="store_true",
                    help="Normalize each pair arrow to a uniform length for visibility.")
    ap.add_argument("--animate_arrow_scale", type=float, default=0.35,
                    help="Relative length for normalized pair arrows (multiplied by PCA scale).")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    noise_shape = parse_noise_shape(args.noise_shape)

    # Load VAE modules
    sd = load_from_standard_weights(args.ckpt, device="cpu")
    encoder = VAE_Encoder(); encoder.load_state_dict(sd["encoder"], strict=True)
    decoder = VAE_Decoder(); decoder.load_state_dict(sd["decoder"], strict=True)
    encoder.eval().to(device); decoder.eval().to(device)

    pairs_dir = Path(args.pairs_dir)
    pairs_csv = pairs_dir / "pairs.csv"
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs.csv not found in {pairs_dir}")

    # 1) Compute global smile axis
    axis, stats = compute_smile_axis_from_pairs(
        encoder=encoder,
        pairs_csv=pairs_csv,
        img_dir=pairs_dir,
        image_size=args.image_size,
        center_crop=args.center_crop,
        norm_to=args.norm_to,
        device=device,
        noise_shape=noise_shape,
        batch=args.batch,
        per_pair_unit=args.per_pair_unit,
    )

    # Save axis & meta
    torch.save({"axis": axis.cpu(), "dim": axis.numel(), "pairs_dir": str(pairs_dir)}, args.save_axis)
    meta = {
        "ckpt": str(args.ckpt),
        "pairs_dir": str(pairs_dir),
        "image_size": args.image_size,
        "center_crop": args.center_crop,
        "norm_to": args.norm_to,
        "per_pair_unit": args.per_pair_unit,
        "stats": stats,
        "save_axis": args.save_axis,
        "noise_shape": args.noise_shape,
    }
    with open(args.meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved axis to {args.save_axis} (dim={axis.numel()}); meta -> {args.meta_json}")
    print(f"Stats: {stats}")

    # 2) Visualization
    if args.plot3d:
        plot_axis_3d_from_pairs(
            encoder=encoder,
            pairs_csv=pairs_csv,
            img_dir=pairs_dir,
            axis=axis.to(device),
            image_size=args.image_size,
            center_crop=args.center_crop,
            norm_to=args.norm_to,
            device=device,
            noise_shape=noise_shape,
            out_png=Path(args.plot3d),
            max_pairs=300,
            plot_pair_arrows=args.plot_pair_arrows,
            max_pair_arrows=args.max_pair_arrows,
        )

    if args.animate3d:
        animate_axis_3d_from_pairs(
            encoder=encoder,
            pairs_csv=pairs_csv,
            img_dir=pairs_dir,
            axis=axis.to(device),
            image_size=args.image_size,
            center_crop=args.center_crop,
            norm_to=args.norm_to,
            device=device,
            noise_shape=noise_shape,
            out_vid=Path(args.animate3d),
            max_pairs=300,
            seconds=args.anim_seconds,
            fps=args.anim_fps,
            elev=args.anim_elev,
            azim_start=args.anim_azim_start,
            spin_degrees=args.anim_spin_degrees,
            animate_pair_arrows=args.animate_pair_arrows,
            max_animate_pair_arrows=args.max_animate_pair_arrows,
            animate_arrow_uniform=args.animate_arrow_uniform,
            animate_arrow_scale=args.animate_arrow_scale,
        )

    # 3) Optional application to folder
    if args.apply_in:
        apply_axis_to_folder(
            encoder=encoder,
            decoder=decoder,
            axis=axis,
            in_dir=Path(args.apply_in),
            out_dir=Path(args.apply_out),
            alphas=args.alphas,
            image_size=args.image_size,
            center_crop=args.center_crop,
            norm_to=args.norm_to,
            device=device,
            noise_shape=noise_shape,
            tangent=(not args.linear_move),
            grid=True
        )
        print(f"[OK] Edited images saved to: {args.apply_out}")


if __name__ == "__main__":
    main()
