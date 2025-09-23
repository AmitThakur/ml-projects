# ldm_cifar_10_cond.py — Conditional Latent Diffusion on CIFAR-10
# Annotated end-to-end training + sampling script with:
#   • Stage A: β‑VAE (32x32x3 -> 8x8x4 latents)
#   • Stage B: Class‑conditional DDPM in latent space
#   • Classifier‑Free Guidance (CFG)
#   • DDIM sampler (fast, typically 50 steps) alongside the default DDPM sampler
#
# Usage examples:
#   Train VAE:     python ldm_cifar_10_cond.py --train-vae --epochs-vae 60
#   Train LDM:     python ldm_cifar_10_cond.py --train-ddpm --epochs-ddpm 200
#   Sample DDIM:   python ldm_cifar_10_cond.py --sample --ckpt runs/.../ldm.pt --sampler ddim --steps 50 --label cat --cfg 3.0
#   Sample DDPM:   python ldm_cifar_10_cond.py --sample --ckpt runs/.../ldm.pt --sampler ddpm --label truck --cfg 3.0

import math, argparse, os, time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid


# Config
@dataclass
class Cfg:
    img_size: int = 32
    nc: int = 3
    num_classes: int = 10

    # VAE (latent autoencoder)
    vae_base: int = 64
    z_ch: int = 4            # latent channels -> target: [B, 4, 8, 8]
    down_factor: int = 4     # 32/4 = 8 latent spatial size

    beta_kl: float = 1.0     # β‑VAE weight on KL term
    lr_vae: float = 2e-4
    bs_vae: int = 256
    epochs_vae: int = 60

    # Latent UNet (for diffusion in latent space)
    dim: int = 192           # base channels (latent is easy; can push to 256)
    dim_mults = (1, 2, 2)    # resolutions: 8 → 4 → 2
    time_dim: int = 256
    label_dim: int = 128
    dropout: float = 0.1
    train_bs: int = 256
    lr_ddpm: float = 2e-4
    epochs_ddpm: int = 200
    T: int = 1000            # total diffusion steps for the base schedule
    p_uncond: float = 0.1    # classifier‑free guidance drop prob

    # EMA
    ema: float = 0.999

    # IO / device
    num_workers: int = 4
    data_root: str = "./data"
    out_root: str = "./runs"
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# Data (CIFAR‑10)
def get_loader(bs, cfg: Cfg):
    tfm = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    ds = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg.num_workers,
                      pin_memory=True, drop_last=True), ds.classes


# VAE (β‑VAE with 8x8 latents)
class Encoder(nn.Module):
    def __init__(self, nc, base, z_ch):
        super().__init__()
        # 32 -> 16 -> 8 feature extractor
        self.net = nn.Sequential(
            nn.Conv2d(nc, base, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base, base, 4, 2, 1), nn.SiLU(),      # 32 -> 16
            nn.Conv2d(base, base*2, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base*2, base*2, 4, 2, 1), nn.SiLU(),  # 16 -> 8
            nn.Conv2d(base*2, base*4, 3, 1, 1), nn.SiLU(),
        )
        # Posterior parameters
        self.mu = nn.Conv2d(base*4, z_ch, 1)
        self.logvar = nn.Conv2d(base*4, z_ch, 1)

    def forward(self, x):
        h = self.net(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = (0.5 * logvar).exp()
        # Reparameterization trick
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_ch, base, nc):
        super().__init__()
        c = base * 4
        self.inp = nn.Conv2d(z_ch, c, 1)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(c, base*2, 4, 2, 1), nn.SiLU(),  # 8 -> 16
            nn.Conv2d(base*2, base*2, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.SiLU(),  # 16 -> 32
            nn.Conv2d(base, base, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base, nc, 3, 1, 1),
        )

    def forward(self, z):
        h = self.inp(z)
        x = self.net(h)
        return torch.tanh(x)

class VAE(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.enc = Encoder(cfg.nc, cfg.vae_base, cfg.z_ch)
        self.dec = Decoder(cfg.z_ch, cfg.vae_base, cfg.nc)

    def forward(self, x):
        z, mu, logvar = self.enc(x)
        xrec = self.dec(z)
        return xrec, mu, logvar, z

# VAE loss (maps to ELBO with L1 recon + β · KL)
def loss_vae(x, xrec, mu, logvar, beta=1.0):
    rec = F.l1_loss(xrec, x)  # reconstruction term
    # KL
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.mean()
    return rec + beta * kl, rec, kl


# Diffusion in latent space
# --------------------------------
# Positional encoding (sinusoidal) for time t
def sinusoidal_time_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLM(nn.Module):

    def __init__(self, in_dim, ch):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, ch * 2))

    def forward(self, e):
        s, b = self.net(e).chunk(2, dim=1)
        return s, b


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, emb_dim, drop=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.film = FiLM(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s,b = self.film(e)  # FiLM(scale, shift) from time+label embedding
        h = h * (1 + s[:, :, None, None]) + b[:, :, None, None]  # ← feature modulation
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Down(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x, e):
        return self.op(x)


class Up(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, 2, 1)

    def forward(self, x, e):
        return self.op(x)


class LatentUNet(nn.Module):

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.inp = nn.Conv2d(cfg.z_ch, cfg.dim, 3, 1, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.time_dim*2), nn.SiLU(), nn.Linear(cfg.time_dim * 2, cfg.time_dim)
        )
        self.label_emb = nn.Embedding(cfg.num_classes+1, cfg.label_dim)  # +1 for null token (CFG)
        self.emb_dim = cfg.time_dim + cfg.label_dim

        in_ch = cfg.dim
        self.downs = nn.ModuleList()
        for i, dm in enumerate(cfg.dim_mults):
            out = cfg.dim*dm
            self.downs += [ResBlock(in_ch, out, self.emb_dim, cfg.dropout),
                           ResBlock(out, out, self.emb_dim, cfg.dropout)]
            if i < len(cfg.dim_mults)-1:
                self.downs += [Down(out)]
            in_ch = out

        self.mid1 = ResBlock(in_ch, in_ch, self.emb_dim,cfg.dropout)
        self.mid2 = ResBlock(in_ch, in_ch, self.emb_dim,cfg.dropout)

        self.ups = nn.ModuleList()
        L = len(cfg.dim_mults)
        for i, dm in reversed(list(enumerate(cfg.dim_mults))):
            out = cfg.dim * dm
            if i == L - 1:
                # bottom: no skip concat
                self.ups += [ResBlock(in_ch, out, self.emb_dim, cfg.dropout)]
            else:
                # higher levels: concat with matching skip (channels = out)
                self.ups += [ResBlock(in_ch + out, out, self.emb_dim, cfg.dropout)]
            self.ups += [ResBlock(out, out, self.emb_dim, cfg.dropout)]
            if i > 0:
                self.ups += [Up(out)]
            in_ch = out

        self.out_n = nn.GroupNorm(8, in_ch)
        self.out_c = nn.Conv2d(in_ch, cfg.z_ch, 3, 1, 1)  # predict ε in latent space

    def embed(self, t, y, cfg: Cfg):
        t = sinusoidal_time_embed(t.float(), cfg.time_dim)
        t = self.time_mlp(t)
        ly = self.label_emb(y)
        return torch.cat([t, ly], dim=1)

    def forward(self, zt, t, y, cfg: Cfg):
        e = self.embed(t, y, cfg)
        h = self.inp(zt)

        L = len(cfg.dim_mults)
        hs = []

        # ----- Down path -----
        it = iter(self.downs)
        for i in range(L):
            h = next(it)(h, e)
            h = next(it)(h, e)
            if i < L - 1:
                hs.append(h)
                h = next(it)(h, e)

        # Middle
        h = self.mid1(h, e)
        h = self.mid2(h, e)

        # ----- Up path -----
        iu = iter(self.ups)
        for i in reversed(range(L)):
            if i < L - 1:
                h = torch.cat([h, hs.pop()], dim=1)
            h = next(iu)(h, e)
            h = next(iu)(h, e)
            if i > 0:
                h = next(iu)(h, e)

        return self.out_c(F.silu(self.out_n(h)))


# Cosine schedule and q(z_t|z_0) sampler (forward process)
@torch.no_grad()
def cosine_beta_schedule(T, s=0.008):
    x = torch.linspace(0, T, T+1)
    ac = torch.cos(((x/T)+s)/(1+s)*math.pi/2)**2
    ac = ac/ac[0]
    betas = 1 - (ac[1:]/ac[:-1])
    return betas.clamp(1e-4, 0.999)


class Diffusion:

    def __init__(self, T, device):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1 - betas
        self.alphas = alphas
        self.ac = torch.cumprod(alphas, dim=0)
        self.ac_prev = F.pad(self.ac[:-1], (1,0), value=1.0)
        self.sqrt_ac = torch.sqrt(self.ac)
        self.sqrt_one_minus_ac = torch.sqrt(1 - self.ac)
        # Posterior variance for DDPM sampling
        self.posterior_var = betas * (1 - self.ac_prev) / (1 - self.ac)

    def q_sample(self, z0, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(z0)
        a = self.sqrt_ac[t][:, None, None, None]
        b = self.sqrt_one_minus_ac[t][:, None, None, None]
        return (a * z0) + (b * noise)


# EMA (for sharper samples)
class EMA:

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for v in self.shadow.values():
            v.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1-self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# Train VAE (Stage A)
def train_vae(cfg: Cfg):
    loader, classes = get_loader(cfg.bs_vae, cfg)
    vae = VAE(cfg).to(cfg.device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.lr_vae)
    out_dir = Path(cfg.out_root) / f"vae_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(cfg.epochs_vae):
        for x,_ in loader:
            x = x.to(cfg.device, non_blocking=True)
            xrec, mu, logv, z = vae(x)
            loss, rec, kl = loss_vae(x, xrec, mu, logv, beta=cfg.beta_kl)  # ELBO mapping
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        # quick reconstruction grid for monitoring
        with torch.no_grad():
            grid = make_grid(torch.cat([x[:16], xrec[:16]], 0), nrow=16, normalize=True, value_range=(-1,1))
            save_image(grid, out_dir/f"ep_{ep:03d}.png")
        print(f"[VAE] epoch {ep} loss={loss.item():.4f} rec={rec.item():.4f} kl={kl.item():.4f}")

    torch.save({"state": vae.state_dict(), "classes": classes, "cfg": cfg.__dict__}, out_dir/"vae.pt")
    print("VAE saved to", out_dir)


# Train latent DDPM (Stage B; VAE frozen)
def train_ddpm(cfg: Cfg, vae_ckpt: Path|None):
    # Load trained VAE
    if vae_ckpt is None:
        vae_ckpt = sorted(Path(cfg.out_root).glob("vae_*/vae.pt"))[-1]
    state = torch.load(vae_ckpt, map_location=cfg.device)
    vae = VAE(cfg).to(cfg.device)
    vae.load_state_dict(state["state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    loader, classes = get_loader(cfg.train_bs, cfg)
    model = LatentUNet(cfg).to(cfg.device)
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_ddpm)
    dif = Diffusion(cfg.T, cfg.device)

    out_dir = Path(cfg.out_root) / f"ldm_{time.strftime('%Y%m%d-%H%M%S')}"
    (out_dir/"samples").mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler(device="cuda", enabled=(cfg.device=="cuda"))

    for ep in range(cfg.epochs_ddpm):
        for x, y in loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            with torch.no_grad():
                z0, mu, logv = vae.enc(x)

            # Classifier‑Free Guidance (train‑time drop)
            y_cfg = y.clone()
            drop = torch.rand_like(y.float()) < cfg.p_uncond
            y_cfg[drop] = cfg.num_classes  # null token index = num_classes

            # Sample timestep and add noise
            t = torch.randint(0, cfg.T, (x.size(0),), device=cfg.device, dtype=torch.long)
            noise = torch.randn_like(z0)
            zt = dif.q_sample(z0, t, noise)

            # Predict noise and regress to true noise (MSE)
            with torch.amp.autocast(device_type="cuda", enabled=(cfg.device=="cuda")):
                eps = model(zt, t, y_cfg, cfg)
                loss = F.mse_loss(eps, noise)  # ← DDPM objective

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ema.update(model)

        # Balanced sample grid each epoch (fast visual check)
        with torch.no_grad():
            imgs = sample_images(model, ema, vae, dif, cfg, classes, n=100, cfg_scale=3.0, sampler="ddim", steps=50)
            save_image(imgs, out_dir/"samples"/f"ep_{ep:03d}.png")
        print(f"[DDPM] epoch {ep} loss={loss.item():.4f}")

    torch.save({"unet": model.state_dict(), "ema": ema.shadow, "vae": state, "classes": classes, "cfg": cfg.__dict__},
               out_dir/"ldm.pt")
    print("LDM saved to", out_dir)



# Sampling utilities: DDPM and DDIM
@torch.no_grad()
def _predict_x0_from_eps(z_t, eps, ac_t):
    return (z_t - torch.sqrt(1 - ac_t) * eps) / torch.sqrt(ac_t)

@torch.no_grad()
def _apply_ema(model, ema, cfg: Cfg):
    # Copy EMA weights into a fresh model used only for inference
    m = LatentUNet(cfg).to(cfg.device)
    if ema is not None:
        ema.copy_to(m)
    else:
        m.load_state_dict(model.state_dict())
    m.eval()
    return m

@torch.no_grad()
def sample_latents_ddpm(model, ema, dif: Diffusion, cfg: Cfg, labels, cfg_scale=3.0):
    """
    Classic DDPM ancestral sampling using posterior variance.
    Maps to the mean/variance update derived from ε‑parameterization.
    """
    m = _apply_ema(model, ema, cfg)
    B = labels.size(0)
    z = torch.randn(B, cfg.z_ch, cfg.img_size//cfg.down_factor, cfg.img_size//cfg.down_factor, device=cfg.device)
    for i in reversed(range(dif.T)):
        t = torch.full((B,), i, device=cfg.device, dtype=torch.long)
        y_un = torch.full_like(labels, cfg.num_classes)
        eps_u = m(z, t, y_un, cfg)
        eps_c = m(z, t, labels, cfg)
        eps = eps_u + cfg_scale*(eps_c - eps_u)

        ac_t = dif.ac[t][:, None, None, None]
        x0 = _predict_x0_from_eps(z, eps, ac_t)
        if i == 0:
            z = torch.sqrt(ac_t) * x0
        else:
            ac_prev = dif.ac_prev[t][:, None, None, None]
            post_var = dif.posterior_var[t][:, None, None, None]
            mean = torch.sqrt(ac_prev) * x0 + torch.sqrt(1 - ac_prev - post_var) * eps
            z = mean + torch.sqrt(post_var) * torch.randn_like(z)
    return z

@torch.no_grad()
def make_ddim_timesteps(T, steps):
    """
    Uniformly strided DDIM schedule: indices in [0..T-1].
    Example: T=1000, steps=50 -> roughly every 20 steps.
    """
    # Ensure we hit t=0
    seq = torch.linspace(0, T-1, steps).long()
    return seq

@torch.no_grad()
def sample_latents_ddim(model, ema, dif: Diffusion, cfg: Cfg, labels, cfg_scale=3.0, steps: int = 50, eta: float = 0.0):
    """
    Deterministic (eta=0) or stochastic (eta>0) DDIM sampler.
    """
    m = _apply_ema(model, ema, cfg)
    B = labels.size(0)
    z = torch.randn(B, cfg.z_ch, cfg.img_size//cfg.down_factor, cfg.img_size//cfg.down_factor, device=cfg.device)

    # Build reduced timestep sequence and helper tensors from the base schedule
    seq = make_ddim_timesteps(dif.T, steps).to(cfg.device)

    for idx in reversed(range(len(seq))):
        t = seq[idx]
        t_prev = seq[idx-1] if idx > 0 else torch.tensor(0, device=cfg.device)
        t_b = torch.full((B,), t, device=cfg.device, dtype=torch.long)

        # CFG: ε̂ = ε̂_u + s(ε̂_c − ε̂_u)
        y_un = torch.full_like(labels, cfg.num_classes)
        eps_u = m(z, t_b, y_un, cfg)
        eps_c = m(z, t_b, labels, cfg)
        eps = eps_u + cfg_scale * (eps_c - eps_u)

        ac_t = dif.ac[t_b][:,None,None,None]
        ac_prev = dif.ac[t_prev.repeat(B)][:, None, None, None]
        x0 = _predict_x0_from_eps(z, eps, ac_t)

        if idx == 0:
            z = torch.sqrt(ac_prev) * x0
            break

        # DDIM noise level (η=0 -> deterministic)
        sigma = eta * torch.sqrt((1 - ac_prev)/(1 - ac_t) * (1 - ac_t/ac_prev))
        # Deterministic direction term
        dir_term = torch.sqrt(1 - ac_prev - sigma**2) * eps
        z = torch.sqrt(ac_prev) * x0 + dir_term + sigma * torch.randn_like(z)

    return z

@torch.no_grad()
def sample_images(model, ema, vae, dif, cfg: Cfg, classes, n=100, cfg_scale=3.0, sampler: str = "ddim", steps: int = 50, eta: float = 0.0, label_name: str|None = None):
    # Build label vector: either balanced grid or single class
    if label_name is None:
        labels = torch.tensor([i for i in range(cfg.num_classes) for _ in range(n//cfg.num_classes)],
                              device=cfg.device, dtype=torch.long)
    else:
        lid = classes.index(label_name)
        labels = torch.full((n,), lid, device=cfg.device, dtype=torch.long)

    # Run chosen sampler in latent space
    if sampler == "ddpm":
        z = sample_latents_ddpm(model, ema, dif, cfg, labels, cfg_scale=cfg_scale)
    elif sampler == "ddim":
        z = sample_latents_ddim(model, ema, dif, cfg, labels, cfg_scale=cfg_scale, steps=steps, eta=eta)
    else:
        raise ValueError("sampler must be 'ddpm' or 'ddim'")

    # Decode through VAE
    x = vae.dec(z)
    grid = make_grid(x, nrow=int(math.sqrt(n)), normalize=True, value_range=(-1, 1))
    return grid


# CLI utils
def sample_cli(cfg: Cfg, ckpt_path: Path, label: str, n: int, cfg_scale: float, sampler: str, steps: int, eta: float):
    state = torch.load(ckpt_path, map_location=cfg.device)
    classes = state["classes"]
    # VAE
    vae = VAE(cfg).to(cfg.device)
    vae.load_state_dict(state["vae"]["state"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad=False
    # UNet
    unet = LatentUNet(cfg).to(cfg.device)
    unet.load_state_dict(state["unet"])
    ema = EMA(unet, cfg.ema); ema.shadow = state["ema"]
    dif = Diffusion(cfg.T, cfg.device)

    grid = sample_images(unet, ema, vae, dif, cfg, classes, n=n, cfg_scale=cfg_scale,
                         sampler=sampler, steps=steps, eta=eta, label_name=label)
    out = Path(ckpt_path).parent/"samples"/f"{label}_n{n}_{sampler}_steps{steps}_cfg{cfg_scale:.1f}_eta{eta:.2f}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out)
    print("Saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-vae", action="store_true")
    ap.add_argument("--train-ddpm", action="store_true")
    ap.add_argument("--sample", action="store_true")

    ap.add_argument("--epochs-vae", type=int, default=None)
    ap.add_argument("--epochs-ddpm", type=int, default=None)

    ap.add_argument("--vae-ckpt", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)

    ap.add_argument("--label", type=str, default="cat")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--cfg", type=float, default=3.0)

    # Sampler controls
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim","ddpm"], help="Sampling algorithm")
    ap.add_argument("--steps", type=int, default=50, help="Sampling steps for DDIM (ignored for DDPM)")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 = deterministic)")

    args = ap.parse_args()

    cfg = Cfg()
    if args.epochs_vae is not None: cfg.epochs_vae = args.epochs_vae
    if args.epochs_ddpm is not None: cfg.epochs_ddpm = args.epochs_ddpm

    if args.train_vae:
        train_vae(cfg); return

    if args.train_ddpm:
        vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else None
        train_ddpm(cfg, vae_ckpt); return

    if args.sample:
        assert args.ckpt is not None, "Provide --ckpt path to ldm.pt"
        sample_cli(cfg, Path(args.ckpt), args.label, args.n, args.cfg, args.sampler, args.steps, args.eta)
        return

    print("Nothing to do. Use --train-vae / --train-ddpm / --sample.")


if __name__ == "__main__":
    main()
