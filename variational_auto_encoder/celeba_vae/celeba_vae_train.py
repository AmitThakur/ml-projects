import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm


# Hyperparameters / Config
DATA_ROOT   = "./data"
RESULTS_DIR = "./results_celeba_vae"
IMG_SIZE    = 128

BATCH_SIZE  = 256
LATENT_DIM  = 256                          # latent vector size
LR          = 2e-4
EPOCHS      = 40
NUM_WORKERS = 0
BETA        = 1.0                          # >1.0 -> more disentangled (β-VAE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)


# CelebA dataset (128x128)
transform = transforms.Compose([
    transforms.CenterCrop(178),            # standard CelebA crop
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                 # [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],  # -> [-1, 1]
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
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.same_channels = (ch_in == ch_out)

        self.block = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),

            nn.GroupNorm(32, ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

        if not self.same_channels:
            self.skip = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.res = ResBlock(ch_in, ch_out)
        self.down = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.res(x)
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.res = ResBlock(ch_out, ch_out)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.res(x)
        return x


# Deep Conv VAE (vector latent)
# 3x128x128 -> 512x8x8 -> 256-d z
class DeepConvVAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        base_ch = 64

        # Encoder
        self.enc_in = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_ch, base_ch)        # 128 -> 64
        self.down2 = DownBlock(base_ch, base_ch * 2)    # 64 -> 32
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)  # 32 -> 16
        self.down4 = DownBlock(base_ch * 4, base_ch * 8)  # 16 -> 8  (512ch)

        self.enc_mid = nn.Sequential(
            ResBlock(base_ch * 8, base_ch * 8),
            ResBlock(base_ch * 8, base_ch * 8),
        )

        # final spatial size is 8x8 with 512 channels
        enc_feat_dim = base_ch * 8 * 8 * 8

        self.fc_mu = nn.Linear(enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, enc_feat_dim)

        self.dec_mid = nn.Sequential(
            ResBlock(base_ch * 8, base_ch * 8),
            ResBlock(base_ch * 8, base_ch * 8),
        )

        self.up1 = UpBlock(base_ch * 8, base_ch * 4)  # 8 -> 16
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)  # 16 -> 32
        self.up3 = UpBlock(base_ch * 2, base_ch)      # 32 -> 64
        self.up4 = UpBlock(base_ch, base_ch)          # 64 -> 128

        self.dec_out = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def encode(self, x):
        x = self.enc_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.enc_mid(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        # reshape to (B, 512, 8, 8)
        base_ch = 64
        x = x.view(x.size(0), base_ch * 8, 8, 8)

        x = self.dec_mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.dec_out(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z



# Loss: β-VAE (MSE + β·KL)
def beta_vae_loss(x_recon, x, mu, logvar, beta=1.0):
    # Reconstruction in image space ([-1,1])
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL(q(z|x) || N(0,I))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl.mean()  # average over batch and latent dims

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def denormalize(x):
    # [-1,1] -> [0,1] for visualization
    return (x * 0.5) + 0.5


def save_reconstructions(model, data_loader, epoch):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(data_loader))
        x = x.to(DEVICE)
        x_recon, _, _, _ = model(x)
        comp = torch.cat([x[:8], x_recon[:8]])
        comp = denormalize(comp).clamp(0, 1)
        utils.save_image(
            comp.cpu(),
            os.path.join(RESULTS_DIR, f"recon_epoch_{epoch:03d}.png"),
            nrow=8
        )


def save_samples(model, epoch, num=64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num, LATENT_DIM).to(DEVICE)
        samples = model.decode(z)
        samples = denormalize(samples).clamp(0, 1)
        utils.save_image(
            samples.cpu(),
            os.path.join(RESULTS_DIR, f"samples_epoch_{epoch:03d}.png"),
            nrow=8
        )


def train():
    model = DeepConvVAE(img_channels=3, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        # tqdm progress bar over batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, _ in pbar:
            x = x.to(DEVICE)
            optimizer.zero_grad()

            x_recon, mu, logvar, _ = model(x)
            loss, recon, kl = beta_vae_loss(x_recon, x, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss  += loss.item()  * bs
            total_recon += recon.item() * bs
            total_kl    += kl.item()    * bs

            # live stats in the progress bar
            pbar.set_postfix(
                loss=loss.item(),
                recon=recon.item(),
                kl=kl.item()
            )

        n = len(train_loader.dataset)
        avg_loss  = total_loss / n
        avg_recon = total_recon / n
        avg_kl    = total_kl / n

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {avg_loss:.5f} | "
            f"Recon: {avg_recon:.5f} | "
            f"KL: {avg_kl:.5f}"
        )

        # periodic visuals + checkpoint
        if epoch == 1 or epoch % 5 == 0:
            save_reconstructions(model, val_loader, epoch)
            save_samples(model, epoch)
            torch.save(
                model.state_dict(),
                os.path.join(RESULTS_DIR, f"celeba_vae_epoch_{epoch:03d}.pth")
            )



if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Train size:", len(train_dataset), "Val size:", len(val_dataset))
    train()
