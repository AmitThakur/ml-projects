"""
python train_celeba_ldm_smile.py \
  --data_root /home/jovyan/data/celeba \
  --out_dir ldm_smile_out \
  --image_size 128 \
  --batch_size 16 \
  --lr 1e-4 \
  --num_steps 1000 \
  --epochs 40 \
  --max_train_images 100000 \
  --max_valid_images 100 \
  --guidance_scale 2.0 \
  --resume

"""
import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DModel


# ---------------------------------------------------------
# 1. Hyperparameters
# ---------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to CelebA root (folder that contains img_align_celeba etc.)")
    parser.add_argument("--out_dir", type=str, default="ldm_smile_out")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_steps", type=int, default=1000, help="Diffusion steps T")
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)

    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--p_uncond", type=float, default=0.1,
                        help="Classifier-free guidance dropout prob during training")

    parser.add_argument("--num_sample_images", type=int, default=16,
                        help="how many images to generate per epoch per class")

    parser.add_argument("--max_train_images", type=int, default=-1,
                        help="max number of training images to use (-1 = all)")
    parser.add_argument("--max_valid_images", type=int, default=-1,
                        help="max number of validation images to use (-1 = all)")

    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in out_dir/checkpoints")

    args = parser.parse_args()
    return args


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_CHANNELS = 4
VAE_SCALE_FACTOR = 0.18215  # SD-style scaling
NUM_CLASSES = 3  # neutral, smiling, null


# ---------------------------------------------------------
# 2. Diffusion schedule helper
# ---------------------------------------------------------
class DiffusionSchedule:
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.to(device)

    def to(self, device):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))


# ---------------------------------------------------------
# 3. Offline CelebA smiling dataset wrapper
# ---------------------------------------------------------
class CelebASmile(Dataset):
    """
    Offline CelebA reader.
    - Reads img_align_celeba, list_attr_celeba.txt, list_eval_partition.txt
    - split ∈ {"train","valid","test"} mapped via partition file (0,1,2)
    - Returns (img, label) with label=0 (non-smiling) or 1 (smiling)
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        super().__init__()
        self.root = os.path.expanduser(root)

        img_dir = os.path.join(self.root, "img_align_celeba")
        attr_path = os.path.join(self.root, "list_attr_celeba.txt")
        part_path = os.path.join(self.root, "list_eval_partition.txt")

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"img_align_celeba not found at {img_dir}")
        if not os.path.isfile(attr_path):
            raise RuntimeError(f"list_attr_celeba.txt not found at {attr_path}")
        if not os.path.isfile(part_path):
            raise RuntimeError(f"list_eval_partition.txt not found at {part_path}")

        # partition file: each line "000001.jpg 0"
        part_dict = {}
        with open(part_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, pid = line.split()
                part_dict[fname] = int(pid)

        # attributes file
        with open(attr_path, "r") as f:
            lines = f.readlines()
        # line 0: num_images
        # line 1: attr names
        attr_names = lines[1].split()
        self.attr_names = attr_names

        if "Smiling" not in attr_names:
            raise RuntimeError("Smiling attribute not found in CelebA attributes.")
        self.smile_idx = attr_names.index("Smiling")

        # filter rows by split using partition file
        data = []
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            fname = parts[0]
            attrs = [int(x) for x in parts[1:]]

            pid = part_dict.get(fname, 0)
            if split == "train" and pid != 0:
                continue
            if split == "valid" and pid != 1:
                continue
            if split == "test" and pid != 2:
                continue

            data.append((fname, torch.tensor(attrs, dtype=torch.int64)))

        self.img_dir = img_dir
        self.samples = data

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {self.root}")

        # image transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),  # -> [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fname, attrs = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        smiling_flag = int(attrs[self.smile_idx].item())  # +1 or -1
        label = 1 if smiling_flag == 1 else 0
        return img, label


# ---------------------------------------------------------
# 4. Helper: VAE encode/decode
# ---------------------------------------------------------
@torch.no_grad()
def encode_vae(vae, x):
    """
    x: [B,3,H,W] in [-1,1]
    returns z0: [B,4,H/8,W/8] scaled.
    Compatible with newer diffusers AutoencoderKL.
    """
    enc_out = vae.encode(x)
    if hasattr(enc_out, "latent_dist"):
        latents = enc_out.latent_dist.sample()
    elif hasattr(enc_out, "sample"):
        latents = enc_out.sample()
    else:
        latents = enc_out
    z0 = latents * VAE_SCALE_FACTOR
    return z0


@torch.no_grad()
def decode_vae(vae, z):
    """
    z: [B,4,h,w] scaled
    returns x in [0,1]
    """
    z_unscaled = z / VAE_SCALE_FACTOR
    dec_out = vae.decode(z_unscaled)
    x = dec_out.sample  # [-1,1]
    x = (x.clamp(-1, 1) + 1) / 2       # -> [0,1]
    return x


# ---------------------------------------------------------
# 5. Conditioning helper: add label as extra channels
# ---------------------------------------------------------
def add_label_channels(z, y, num_classes=NUM_CLASSES):
    """
    z: [B, C, H, W] latent
    y: [B] int labels in [0, num_classes-1]
    Returns: [B, C+num_classes, H, W] with one-hot label broadcast over spatial dims.
    """
    B, C, H, W = z.shape
    y_onehot = F.one_hot(y, num_classes=num_classes).float()  # [B, num_classes]
    y_onehot = y_onehot.view(B, num_classes, 1, 1).expand(B, num_classes, H, W)
    z_in = torch.cat([z, y_onehot], dim=1)
    return z_in


# ---------------------------------------------------------
# 6. Sampling: from noise (class-conditional) with CFG
# ---------------------------------------------------------
@torch.no_grad()
def sample_ldm(unet, schedule, vae, num_samples, label, guidance_scale,
               latent_shape, device):
    """
    Sample from LDM with classifier-free guidance.

    label: 0 (neutral) or 1 (smiling)
    Uses null label 2 for uncond path.
    """
    T = schedule.num_steps
    C, H, W = latent_shape

    x = torch.randn(num_samples, C, H, W, device=device)

    y_cond = torch.full((num_samples,), int(label), device=device, dtype=torch.long)
    y_uncond = torch.full((num_samples,), 2, device=device, dtype=torch.long)  # null label

    for t_step in tqdm(reversed(range(T)), desc="DDPM sampling", leave=False):
        t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)

        x_uncond_in = add_label_channels(x, y_uncond)  # [B,4+3,H,W]
        x_cond_in = add_label_channels(x, y_cond)      # [B,4+3,H,W]

        eps_uncond = unet(x_uncond_in, t).sample
        eps_cond = unet(x_cond_in, t).sample

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta_t = schedule.betas[t_step]
        alpha_t = schedule.alphas[t_step]
        alpha_bar_t = schedule.alphas_cumprod[t_step]
        sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t_step]
        sqrt_recip_alpha_t = schedule.sqrt_recip_alphas[t_step]

        x = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alpha_bar_t * eps)

        if t_step > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(schedule.posterior_variance[t_step])
            x = x + sigma_t * noise

    x_img = decode_vae(vae, x)
    return x_img


# ---------------------------------------------------------
# 7. Editing: neutral image -> smiling
# (your current version with t_start_ratio and smaller guidance in call)
# ---------------------------------------------------------
@torch.no_grad()
def edit_smile(unet, schedule, vae, x_neutral, guidance_scale, device,
               t_start_ratio=0.5):
    """
    x_neutral: [B,3,H,W] in [-1,1]
    t_start_ratio: fraction of the diffusion chain to start from (0<r<=1).
    Returns smiling images [B,3,H,W] in [0,1].
    """
    unet.eval()
    vae.eval()

    z0 = encode_vae(vae, x_neutral.to(device))   # [B,4,h,w]
    B, C, H, W = z0.shape
    T = schedule.num_steps

    t_start = max(1, int(t_start_ratio * (T - 1)))
    alpha_bar_t = schedule.alphas_cumprod[t_start]

    eps = torch.randn_like(z0)
    z_t = math.sqrt(alpha_bar_t) * z0 + math.sqrt(1.0 - alpha_bar_t) * eps

    y_smile = torch.full((B,), 1, device=device, dtype=torch.long)
    y_null = torch.full((B,), 2, device=device, dtype=torch.long)

    for t_step in tqdm(reversed(range(t_start + 1)), desc="Smile editing", leave=False):
        t = torch.full((B,), t_step, device=device, dtype=torch.long)

        z_uncond_in = add_label_channels(z_t, y_null)
        z_cond_in = add_label_channels(z_t, y_smile)

        eps_uncond = unet(z_uncond_in, t).sample
        eps_cond = unet(z_cond_in, t).sample
        eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta_t = schedule.betas[t_step]
        alpha_t = schedule.alphas[t_step]
        alpha_bar_t = schedule.alphas_cumprod[t_step]
        sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t_step]
        sqrt_recip_alpha_t = schedule.sqrt_recip_alphas[t_step]

        z_t = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_alpha_bar_t * eps_hat)

        if t_step > 0:
            noise = torch.randn_like(z_t)
            sigma_t = torch.sqrt(schedule.posterior_variance[t_step])
            z_t = z_t + sigma_t * noise

    x_smile = decode_vae(vae, z_t)  # [B,3,H,W] in [0,1]
    return x_smile


# ---------------------------------------------------------
# 8. Plot loss curve
# ---------------------------------------------------------
def plot_losses(losses, out_path):
    epochs = list(range(1, len(losses) + 1))
    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss (MSE)")
    plt.title("Latent Diffusion Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------
# 9. Main training loop (with resume)
# ---------------------------------------------------------
def main():
    args = get_args()
    print("Using device:", DEVICE)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    (ckpt_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "edits").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # 9.1 Datasets + subset limiting (offline CelebA)
    train_set_full = CelebASmile(args.data_root, split="train", image_size=args.image_size)
    valid_set_full = CelebASmile(args.data_root, split="valid", image_size=args.image_size)

    if args.max_train_images > 0:
        n_train = min(args.max_train_images, len(train_set_full))
        train_indices = list(range(n_train))
        train_set = Subset(train_set_full, train_indices)
        print(f"Using {n_train}/{len(train_set_full)} images for training.")
    else:
        train_set = train_set_full
        print(f"Using all {len(train_set_full)} training images.")

    if args.max_valid_images > 0:
        n_valid = min(args.max_valid_images, len(valid_set_full))
        valid_indices = list(range(n_valid))
        valid_set = Subset(valid_set_full, valid_indices)
        print(f"Using {n_valid}/{len(valid_set_full)} images for validation.")
    else:
        valid_set = valid_set_full
        print(f"Using all {len(valid_set_full)} validation images.")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 9.2 VAE (frozen)
    print("Loading pretrained VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(DEVICE)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 9.3 UNet in latent space (label-as-channel conditional)
    sample_size = args.image_size // 8  # VAE downsampling factor 8
    print("Sample (latent) spatial size:", sample_size)

    unet = UNet2DModel(
        sample_size=sample_size,
        in_channels=LATENT_CHANNELS + NUM_CLASSES,  # 4 latent + 3 label channels
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=(256, 512, 512),  # tweak if you want bigger model
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    unet.to(DEVICE)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # 9.4 Diffusion schedule
    schedule = DiffusionSchedule(
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=DEVICE
    )

    # 9.5 Resume logic
    start_epoch = 1
    epoch_losses = []

    if args.resume:
        ckpts = sorted(ckpt_dir.glob("unet_epoch_*.pt"))
        if len(ckpts) == 0:
            print(f"[RESUME] No checkpoints found in {ckpt_dir}, starting from scratch.")
        else:
            latest_ckpt_path = ckpts[-1]
            print(f"[RESUME] Loading checkpoint: {latest_ckpt_path}")
            ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)

            unet.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            epoch_losses = ckpt.get("losses", [])
            last_epoch = ckpt.get("epoch", 0)
            start_epoch = last_epoch + 1
            print(f"[RESUME] Last epoch = {last_epoch}, resuming from epoch {start_epoch}")

    if start_epoch > args.epochs:
        print(f"start_epoch ({start_epoch}) > args.epochs ({args.epochs}). Nothing to train.")
        return

    # 9.6 Training
    for epoch in range(start_epoch, args.epochs + 1):
        unet.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(DEVICE)                  # [B,3,H,W] in [-1,1]
            y = y.to(DEVICE, dtype=torch.long)  # [B] in {0,1}

            B = x.size(0)
            optimizer.zero_grad()

            with torch.no_grad():
                z0 = encode_vae(vae, x)       # [B,4,h,w]

            # sample random t for each element
            t = torch.randint(0, args.num_steps, (B,), device=DEVICE, dtype=torch.long)

            # sample noise
            eps = torch.randn_like(z0)

            alpha_bar_t = schedule.alphas_cumprod[t].view(B, 1, 1, 1)
            z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1.0 - alpha_bar_t) * eps

            # classifier-free guidance training: label dropout
            drop_mask = (torch.rand(B, device=DEVICE) < args.p_uncond)
            y_train = y.clone()
            y_train[drop_mask] = 2  # null label

            z_in = add_label_channels(z_t, y_train)  # [B,4+3,h,w]
            noise_pred = unet(z_in, t).sample
            loss = F.mse_loss(noise_pred, eps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        print(f"===> Epoch {epoch}: avg train loss = {avg_loss:.6f}")

        # Save checkpoint
        ckpt_path = ckpt_dir / f"unet_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": epoch_losses,
        }, ckpt_path)

        # Plot loss curve
        plot_path = out_dir / "plots" / "train_loss.png"
        plot_losses(epoch_losses, plot_path)

        # -------------------------------------------------
        # Sampling at this epoch (from pure noise)
        # -------------------------------------------------
        unet.eval()
        with torch.no_grad():
            num_samples = args.num_sample_images
            latent_shape = (LATENT_CHANNELS, sample_size, sample_size)

            imgs_neutral = sample_ldm(
                unet, schedule, vae,
                num_samples=num_samples,
                label=0,
                guidance_scale=args.guidance_scale,
                latent_shape=latent_shape,
                device=DEVICE
            )

            imgs_smile = sample_ldm(
                unet, schedule, vae,
                num_samples=num_samples,
                label=1,
                guidance_scale=args.guidance_scale,
                latent_shape=latent_shape,
                device=DEVICE
            )

            # stack [neutral ; smiling]
            grid = torch.cat([imgs_neutral, imgs_smile], dim=0)
            sample_path = out_dir / "samples" / f"samples_epoch_{epoch:03d}.png"
            save_image(grid, sample_path, nrow=num_samples, padding=2)

        # -------------------------------------------------
        # Edit step: single random neutral image from validation
        # -------------------------------------------------
        print("Running edit_smile on a random neutral validation image...")

        found = False
        for _ in tqdm(range(50), desc="Searching neutral val image", leave=False):
            idx = random.randint(0, len(valid_set) - 1)
            x_val, y_val = valid_set[idx]
            if y_val == 0:
                found = True
                break

        if not found:
            print("Warning: could not find neutral image in validation subset for editing.")
        else:
            x_neutral = x_val.unsqueeze(0).to(DEVICE)  # [1,3,H,W] in [-1,1]

            x_smiling = edit_smile(
                unet, schedule, vae,
                x_neutral=x_neutral,
                guidance_scale=min(args.guidance_scale, 2.0),  # e.g. softer for editing
                device=DEVICE,
                t_start_ratio=0.25,
            )  # [1,3,H,W] in [0,1]

            # original neutral in [0,1] for visualization
            x_neutral_vis = (x_neutral.clamp(-1, 1) + 1) / 2.0  # [1,3,H,W]

            # side-by-side: [original, edited]
            comparison = torch.cat([x_neutral_vis, x_smiling], dim=0)
            edit_path = out_dir / "edits" / f"edit_epoch_{epoch:03d}.png"
            save_image(comparison, edit_path, nrow=2, padding=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
