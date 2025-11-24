# TODO: Fix it. Saturates to black after epoch 7.
import os
import re
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# Config

DATA_ROOT   = "./data"
RESULTS_DIR = "./ldm_vae_celeba_128"
IMG_SIZE    = 128

BATCH_SIZE   = 64
LATENT_CH    = 4            # spatial latent channels (like SD)
BASE_CH      = 128          # base channel width
LR           = 2e-4
EPOCHS       = 40
NUM_WORKERS  = 0
KL_WEIGHT    = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)


# CelebA Dataset (128x128)
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                     # [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],      # -> [-1,1]
                         [0.5, 0.5, 0.5]),
])

train_dataset = datasets.CelebA(
    root=DATA_ROOT,
    split="train",
    transform=transform,
    download=False
)

val_dataset = datasets.CelebA(
    root=DATA_ROOT,
    split="valid",
    transform=transform,
    download=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=True
)


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, groups=32):
        super().__init__()
        self.same_channels = (ch_in == ch_out)

        self.block = nn.Sequential(
            nn.GroupNorm(groups, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),

            nn.GroupNorm(groups, ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

        if self.same_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        return self.skip(x) + self.block(x)


class SelfAttention2d(nn.Module):
    """
    Simple 1-head self-attention over spatial dims (H*W).
    Used at 16x16 resolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)              # [B, 3C, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # flatten spatial
        q = q.view(b, c, h * w)            # [B, C, HW]
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # attention
        attn = torch.bmm(q.transpose(1, 2), k)  # [B, HW, HW]
        attn = attn / (c ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))  # [B, C, HW]
        out = out.view(b, c, h, w)
        out = self.proj(out)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(ch_in, ch_out)
        self.res2 = ResBlock(ch_out, ch_out)
        self.attn = SelfAttention2d(ch_out) if use_attn else nn.Identity()
        self.down = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out, use_attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.res1 = ResBlock(ch_out, ch_out)
        self.res2 = ResBlock(ch_out, ch_out)
        self.attn = SelfAttention2d(ch_out) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        return x


# LDM-style Autoencoder
# 128x128 -> 4x16x16 latent
class LDMAutoencoder(nn.Module):
    def __init__(self, img_channels=3, base_ch=128, z_channels=4):
        super().__init__()
        self.z_channels = z_channels
        self.base_ch = base_ch

        # Encoder
        self.enc_in = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)

        # 128 -> 64
        self.down1 = DownBlock(base_ch, base_ch, use_attn=False)
        # 64 -> 32
        self.down2 = DownBlock(base_ch, base_ch * 2, use_attn=False)
        # 32 -> 16
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, use_attn=True)  # attn at 16x16

        # mid at 16x16
        mid_ch = base_ch * 4
        self.mid_block1 = ResBlock(mid_ch, mid_ch)
        self.mid_attn = SelfAttention2d(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch)

        # mu, log: conv 3x3, keep spatial 16x16
        self.to_mu = nn.Conv2d(mid_ch, z_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv2d(mid_ch, z_channels, kernel_size=3, padding=1)

        # Decoder
        self.dec_in = nn.Conv2d(z_channels, mid_ch, kernel_size=3, padding=1)

        self.mid_dec_block1 = ResBlock(mid_ch, mid_ch)
        self.mid_dec_attn = SelfAttention2d(mid_ch)
        self.mid_dec_block2 = ResBlock(mid_ch, mid_ch)

        # mirror: 16 -> 32 -> 64 -> 128
        self.up1 = UpBlock(mid_ch, base_ch * 2, use_attn=True)   # 16 -> 32
        self.up2 = UpBlock(base_ch * 2, base_ch, use_attn=False) # 32 -> 64
        self.up3 = UpBlock(base_ch, base_ch, use_attn=False)     # 64 -> 128

        self.dec_out = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # [-1,1]
        )

    def encode(self, x):
        # x: [B,3,128,128]
        x = self.enc_in(x)    # [B,base,128,128]
        x = self.down1(x)     # [B,base,64,64]
        x = self.down2(x)     # [B,2base,32,32]
        x = self.down3(x)     # [B,4base,16,16]

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        mu = self.to_mu(x)         # [B,z_ch,16,16]
        logvar = self.to_logvar(x) # [B,z_ch,16,16]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # deterministic at eval
            return mu

    def decode(self, z):
        x = self.dec_in(z)           # [B,4base,16,16]
        x = self.mid_dec_block1(x)
        x = self.mid_dec_attn(x)
        x = self.mid_dec_block2(x)

        x = self.up1(x)              # -> [B,2base,32,32]
        x = self.up2(x)              # -> [B,base,64,64]
        x = self.up3(x)              # -> [B,base,128,128]

        x = self.dec_out(x)          # -> [B,3,128,128] in [-1,1]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z



# Loss (MSE + small KL)
def ldm_vae_loss(x_recon, x, mu, logvar, kl_weight=1e-4):
    # recon in [-1,1], MSE mean
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL over spatial latent
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl.mean()  # average over batch, channels, H,W

    loss = recon_loss + kl_weight * kl_loss
    return loss, recon_loss, kl_loss


def denormalize(x):
    # [-1,1] -> [0,1]
    return (x * 0.5) + 0.5


def save_reconstructions(model, data_loader, epoch):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(data_loader))
        x = x.to(DEVICE)
        x_recon, _, _, _ = model(x)
        comp = torch.cat([x[:8], x_recon[:8]])
        comp = denormalize(comp).clamp(0, 1)
        path = os.path.join(RESULTS_DIR, f"recon_epoch_{epoch:03d}.png")
        utils.save_image(comp.cpu(), path, nrow=8)
    return path


def save_samples(model, epoch, num=64):
    model.eval()
    with torch.no_grad():
        H = IMG_SIZE // 8
        W = IMG_SIZE // 8
        z = torch.randn(num, LATENT_CH, H, W).to(DEVICE)
        samples = model.decode(z)
        samples = denormalize(samples).clamp(0, 1)
        path = os.path.join(RESULTS_DIR, f"samples_epoch_{epoch:03d}.png")
        utils.save_image(samples.cpu(), path, nrow=8)
    return path


def cleanup_old_files(pattern, keep=5):
    files = glob(pattern)
    if len(files) <= keep:
        return
    # sort by epoch number extracted from filename
    def extract_epoch(fname):
        m = re.search(r"_epoch_(\d+)", os.path.basename(fname))
        return int(m.group(1)) if m else -1
    files_sorted = sorted(files, key=extract_epoch)
    for f in files_sorted[:-keep]:
        try:
            os.remove(f)
        except OSError:
            pass


def plot_losses(history, out_path):
    epochs = range(1, len(history["train_total"]) + 1)
    plt.figure(figsize=(8, 6))

    # total
    plt.plot(epochs, history["train_total"], label="train_total")
    plt.plot(epochs, history["val_total"],   label="val_total",   linestyle="--")

    # recon
    plt.plot(epochs, history["train_recon"], label="train_recon")
    plt.plot(epochs, history["val_recon"],   label="val_recon",   linestyle="--")

    # KL
    plt.plot(epochs, history["train_kl"],    label="train_kl")
    plt.plot(epochs, history["val_kl"],      label="val_kl",      linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Losses (total / recon / KL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# Training loop
def train():
    model = LDMAutoencoder(
        img_channels=3,
        base_ch=BASE_CH,
        z_channels=LATENT_CH
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # for plotting
    history = {
        "train_total": [],
        "train_recon": [],
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, _ in pbar:
            x = x.to(DEVICE)
            optimizer.zero_grad()

            x_recon, mu, logvar, _ = model(x)
            loss, recon, kl = ldm_vae_loss(
                x_recon, x, mu, logvar, kl_weight=KL_WEIGHT
            )
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_total += loss.item() * bs
            train_recon += recon.item() * bs
            train_kl += kl.item() * bs

            pbar.set_postfix(
                loss=loss.item(),
                recon=recon.item(),
                kl=kl.item()
            )

        n_train = len(train_loader.dataset)
        train_total /= n_train
        train_recon /= n_train
        train_kl    /= n_train

        # Validation
        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                x_recon, mu, logvar, _ = model(x)
                loss, recon, kl = ldm_vae_loss(
                    x_recon, x, mu, logvar, kl_weight=KL_WEIGHT
                )
                bs = x.size(0)
                val_total += loss.item() * bs
                val_recon += recon.item() * bs
                val_kl += kl.item() * bs

        n_val = len(val_loader.dataset)
        val_total /= n_val
        val_recon /= n_val
        val_kl /= n_val

        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"Epoch {epoch:03d} | "
            f"Train: total={train_total:.5f}, recon={train_recon:.5f}, kl={train_kl:.5f} | "
            f"Val: total={val_total:.5f}, recon={val_recon:.5f}, kl={val_kl:.5f}"
        )

        recon_path = save_reconstructions(model, val_loader, epoch)
        samples_path = save_samples(model, epoch)

        ckpt_path = os.path.join(RESULTS_DIR, f"ldm_vae_epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        cleanup_old_files(os.path.join(RESULTS_DIR, "ldm_vae_epoch_*.pth"), keep=5)
        cleanup_old_files(os.path.join(RESULTS_DIR, "recon_epoch_*.png"), keep=5)
        cleanup_old_files(os.path.join(RESULTS_DIR, "samples_epoch_*.png"), keep=5)

        plot_losses(history, os.path.join(RESULTS_DIR, "losses.png"))


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Train size:", len(train_dataset), "Val size:", len(val_dataset))
    train()
