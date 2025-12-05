"""
python celeba_ldm_smile.py \
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

python celeba_ldm_smile.py \
  --mode edit \
  --data_root /home/jovyan/data/celeba \
  --out_dir ldm_smile_out \
  --image_size 128 \
  --num_steps 1000 \
  --ckpt_path ldm_smile_out/checkpoints/unet_epoch_300.pt \
  --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --t_start_ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --edit_image_path pairs/0003_neutral.jpg \
  --edit_guidance_scale 2.0 \
  --max_valid_images 500

python celeba_ldm_smile.py \
  --mode edit \
  --data_root /home/jovyan/data/celeba \
  --out_dir ldm_smile_out \
  --image_size 128 \
  --num_steps 1000 \
  --ckpt_path ldm_smile_out/checkpoints/unet_epoch_300.pt \
  --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --t_start_ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --edit_guidance_scale 2.0 \
  --max_valid_images 500

"""
import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DModel




# Hyperparameters / CLI
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "edit"],
                        help="train: train LDM; edit: run alpha-edit from a checkpoint")

    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to CelebA root (folder that contains img_align_celeba etc.)")
    parser.add_argument("--out_dir", type=str, default="ldm_smile_out")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

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

    # ---- edit-only mode args ----
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Checkpoint .pt path for --mode edit")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="Alpha values for label interpolation neutral->smile in edit mode")
    parser.add_argument("--edit_guidance_scale", type=float, default=2.0,
                        help="CFG guidance scale to use in edit mode")
    parser.add_argument(
        "--t_start_ratios", type=float, nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="List of t_start_ratio values for t×alpha grid in --mode edit"
    )
    parser.add_argument(
        "--edit_image_path", type=str, default=None,
        help="Path to input image to edit in --mode edit. If not set, use random neutral val image."
    )

    args = parser.parse_args()
    return args


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_CHANNELS = 4
VAE_SCALE_FACTOR = 0.18215  # SD-style scaling
NUM_CLASSES = 3  # neutral, smiling, null



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


class CelebASmile(Dataset):
    """
    Using it in case gdown is not working for Pytorch dataset download.
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



Label Conditioning: supports hard AND soft labels
def add_label_channels(z, y, num_classes=NUM_CLASSES):
    """
    z: [B, C, H, W] latent
    y:
      - either [B] int labels in [0, num_classes-1]
      - or [B, num_classes] float soft labels (e.g. interpolated neutral/smile)
    Returns: [B, C+num_classes, H, W]
    """
    B, C, H, W = z.shape

    if y.dim() == 1:  # integer labels
        y_onehot = F.one_hot(y, num_classes=num_classes).float()  # [B, num_classes]
    else:
        y_onehot = y.float()  # assume already [B, num_classes]

    y_onehot = y_onehot.view(B, num_classes, 1, 1).expand(B, num_classes, H, W)
    z_in = torch.cat([z, y_onehot], dim=1)
    return z_in



# Sampling: from noise (class-conditional) with CFG
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



# Editing: alpha interpolation on class label
@torch.no_grad()
def edit_smile_alpha(unet, schedule, vae, x_neutral, alphas,
                     guidance_scale, device, t_start_ratio=0.5):
    """
    Edit a neutral image into varying degrees of smile via alpha interpolation in label space.

    x_neutral: [1,3,H,W] in [-1,1]
    alphas:    list or 1D tensor of values in [0,1], e.g. [0.0, 0.25, 0.5, 0.75, 1.0]
    guidance_scale: CFG scale for editing
    t_start_ratio: fraction of diffusion chain to start from (0<r<=1)

    Returns: [len(alphas), 3, H, W] images in [0,1]
    """
    unet.eval()
    vae.eval()

    if isinstance(alphas, list):
        alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
    else:
        alphas = alphas.to(device).float()

    num_alpha = alphas.shape[0]

    # encode once, replicate for each alpha
    z0_single = encode_vae(vae, x_neutral.to(device))  # [1,4,h,w]
    _, C, H, W = z0_single.shape
    z0 = z0_single.expand(num_alpha, C, H, W).clone()  # [A,4,h,w]

    T = schedule.num_steps
    t_start = max(1, int(t_start_ratio * (T - 1)))
    alpha_bar_t = schedule.alphas_cumprod[t_start]

    eps = torch.randn_like(z0)
    z_t = math.sqrt(alpha_bar_t) * z0 + math.sqrt(1.0 - alpha_bar_t) * eps  # [A,4,h,w]

    # soft labels between neutral (index 0) and smile (index 1)
    y_soft = torch.zeros(num_alpha, NUM_CLASSES, device=device)  # [A,3]
    y_soft[:, 0] = 1.0 - alphas
    y_soft[:, 1] = alphas
    # y_soft[:, 2] = 0

    y_null = torch.full((num_alpha,), 2, device=device, dtype=torch.long)

    for t_step in tqdm(reversed(range(t_start + 1)), desc="Smile alpha editing", leave=False):
        t = torch.full((num_alpha,), t_step, device=device, dtype=torch.long)

        z_uncond_in = add_label_channels(z_t, y_null)
        z_cond_in   = add_label_channels(z_t, y_soft)

        eps_uncond = unet(z_uncond_in, t).sample
        eps_cond   = unet(z_cond_in, t).sample
        eps_hat    = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

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

    x_edits = decode_vae(vae, z_t)  # [A,3,H,W] in [0,1]
    return x_edits


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



def run_train(args):
    print("Using device:", DEVICE)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "edits").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # datasets
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

    # VAE
    print("Loading pretrained VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(DEVICE)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # UNet
    sample_size = args.image_size // 8
    print("Latent spatial size:", sample_size)

    unet = UNet2DModel(
        sample_size=sample_size,
        in_channels=LATENT_CHANNELS + NUM_CLASSES,
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=(256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    unet.to(DEVICE)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    schedule = DiffusionSchedule(
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=DEVICE
    )

    # resume
    start_epoch = 1
    epoch_losses = []
    if args.resume:
        ckpts = sorted(ckpt_dir.glob("unet_epoch_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"[RESUME] loading {latest}")
            ckpt = torch.load(latest, map_location=DEVICE)
            unet.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            epoch_losses = ckpt.get("losses", [])
            last_epoch = ckpt.get("epoch", 0)
            start_epoch = last_epoch + 1
            print(f"[RESUME] last epoch {last_epoch} -> start {start_epoch}")
        else:
            print("[RESUME] no checkpoints found, starting from scratch")

    if start_epoch > args.epochs:
        print(f"start_epoch ({start_epoch}) > epochs ({args.epochs}); nothing to train.")
        return

    # training loop
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
        torch.save(
            dict(epoch=epoch, model=unet.state_dict(),
                 optimizer=optimizer.state_dict(), losses=epoch_losses),
            ckpt_path
        )

        plot_losses(epoch_losses, out_dir / "plots" / "train_loss.png")

        # sampling
        unet.eval()
        with torch.no_grad():
            latent_shape = (LATENT_CHANNELS, sample_size, sample_size)
            n_samples = args.num_sample_images

            imgs_neutral = sample_ldm(unet, schedule, vae,
                                      num_samples=n_samples, label=0,
                                      guidance_scale=args.guidance_scale,
                                      latent_shape=latent_shape, device=DEVICE)
            imgs_smile = sample_ldm(unet, schedule, vae,
                                    num_samples=n_samples, label=1,
                                    guidance_scale=args.guidance_scale,
                                    latent_shape=latent_shape, device=DEVICE)

            # stack [neutral ; smiling]
            grid = torch.cat([imgs_neutral, imgs_smile], dim=0)
            sample_path = out_dir / "samples" / f"samples_epoch_{epoch:03d}.png"
            save_image(grid, sample_path, nrow=n_samples, padding=2)

        # alpha-edit preview (same as before but inside training)
        print("Running alpha edit preview on random neutral val image...")
        valid_set = valid_set_full if args.max_valid_images <= 0 else Subset(
            valid_set_full, list(range(min(args.max_valid_images, len(valid_set_full))))
        )

        found = False
        for _ in tqdm(range(50), desc="Searching neutral val image", leave=False):
            idx = random.randint(0, len(valid_set) - 1)
            x_val, y_val = valid_set[idx]
            if y_val == 0:
                found = True
                break

        if found:
            x_neutral = x_val.unsqueeze(0).to(DEVICE)
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            x_edits = edit_smile_alpha(
                unet, schedule, vae,
                x_neutral=x_neutral, alphas=alphas,
                guidance_scale=min(args.guidance_scale, 2.0),
                device=DEVICE, t_start_ratio=0.5
            )
            x_neutral_vis = (x_neutral.clamp(-1, 1) + 1) / 2.0
            comparison = torch.cat([x_neutral_vis, x_edits], dim=0)
            save_image(comparison, out_dir / "edits" / f"edit_epoch_{epoch:03d}.png",
                       nrow=len(alphas) + 1, padding=2)
        else:
            print("Warning: no neutral val image found for preview.")

    print("Training complete.")


def run_edit_only(args):
    print("Edit-only mode (t_start_ratio × alpha grid). Device:", DEVICE)

    if args.ckpt_path is None:
        raise ValueError("--ckpt_path must be provided in --mode edit")

    out_dir = Path(args.out_dir)
    (out_dir / "edits").mkdir(parents=True, exist_ok=True)

    # Load VAE
    print("Loading pretrained VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.to(DEVICE)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Load UNET checkpoint
    sample_size = args.image_size // 8
    unet = UNet2DModel(
        sample_size=sample_size,
        in_channels=LATENT_CHANNELS + NUM_CLASSES,
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=(256, 512, 512),  # match your training config
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    unet.to(DEVICE)

    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=DEVICE)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    schedule = DiffusionSchedule(
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=DEVICE
    )



    # Obtain neutral image: user-supplied path OR random neutral from CelebA-valid
    if args.edit_image_path is not None:
        # ---- use user supplied image ----
        print(f"Using user-supplied image: {args.edit_image_path}")
        img = Image.open(args.edit_image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),  # -> [-1,1]
        ])
        x_neutral = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,H,W] in [-1,1]

    else:
        # ---- fall back to random neutral from CelebA valid ----
        print("Selecting random neutral validation image from CelebA...")

        valid_set_full = CelebASmile(args.data_root, split="valid", image_size=args.image_size)
        if args.max_valid_images > 0:
            n_valid = min(args.max_valid_images, len(valid_set_full))
            valid_set = Subset(valid_set_full, list(range(n_valid)))
            print(f"Using {n_valid}/{len(valid_set_full)} validation images.")
        else:
            valid_set = valid_set_full
            print(f"Using all {len(valid_set_full)} validation images.")

        found = False
        for _ in range(200):
            idx = random.randint(0, len(valid_set) - 1)
            x_val, y_val = valid_set[idx]
            if y_val == 0:
                found = True
                break

        if not found:
            raise RuntimeError("Could not find neutral image in validation subset.")

        x_neutral = x_val.unsqueeze(0).to(DEVICE)  # [1,3,H,W] in [-1,1]

    # x_neutral is now [1,3,H,W] in [-1,1]
    x_neutral_vis = (x_neutral.clamp(-1, 1) + 1) / 2.0  # [1,3,H,W] in [0,1]

    alphas = args.alphas
    t_list = args.t_start_ratios
    print(f"Using t_start_ratios = {t_list}, alphas = {alphas}")


    # Run Edits for each (t_start, alpha)
    rows_imgs = []  # list of [num_cols,3,H,W]
    for t_ratio in t_list:
        print(f"  Editing row for t_start_ratio = {t_ratio} ...")
        x_edits = edit_smile_alpha(
            unet, schedule, vae,
            x_neutral=x_neutral,
            alphas=alphas,
            guidance_scale=args.edit_guidance_scale,
            device=DEVICE,
            t_start_ratio=t_ratio,
        )  # [A,3,H,W] in [0,1]

        row_imgs = torch.cat([x_neutral_vis, x_edits], dim=0)  # [K+1,3,H,W]
        rows_imgs.append(row_imgs)


    # Build grid
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except:
        font = ImageFont.load_default()

    num_rows = len(t_list)
    num_cols = len(alphas) + 1  # neutral + edits
    H = args.image_size
    W = args.image_size

    label_h = 30   # top label height
    label_w = 90   # left label width

    canvas_w = label_w + num_cols * W
    canvas_h = label_h + num_rows * H

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column labels
    col_x0 = label_w
    draw.text((col_x0 + W // 2 - 35, 5), "neutral", fill=(0, 0, 0), font=font)
    for j, a in enumerate(alphas):
        col_x = label_w + (j + 1) * W
        label = f"alpha={a:.2f}"
        draw.text((col_x + W // 2 - 35, 5), label, fill=(0, 0, 0), font=font)

    # Row labels
    for i, t_ratio in enumerate(t_list):
        row_y = label_h + i * H + H // 2 - 10
        label = f"t={t_ratio:.2f} T"
        draw.text((5, row_y), label, fill=(0, 0, 0), font=font)

    # Paste grid images
    for i in range(num_rows):
        row_imgs = rows_imgs[i]  # [num_cols,3,H,W]
        for j in range(num_cols):
            img_t = row_imgs[j].cpu().permute(1, 2, 0).numpy()
            img_t = (img_t * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_t)
            x = label_w + j * W
            y = label_h + i * H
            canvas.paste(pil_img, (x, y))

    # Save result
    ckpt_name = Path(args.ckpt_path).stem
    base = f"edit_grid_t_alpha_{ckpt_name}"
    if args.edit_image_path is not None:
        base += "_custom"

    out_path = out_dir / "edits" / f"{base}.png"
    canvas.save(out_path)
    print(f"Saved t_start_ratio × alpha grid to:\n{out_path}")



if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        run_train(args)
    else:  # edit
        run_edit_only(args)

