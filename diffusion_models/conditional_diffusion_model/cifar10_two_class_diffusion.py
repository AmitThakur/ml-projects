"""
Two-class conditional DDPM on CIFAR-10 (cats vs dogs)
-----------------------------------------------------
- Trains a class-conditional diffusion model on ONLY two classes from CIFAR-10.
- Cosine noise schedule, EMA, classifier-free guidance, loss plotting, and optional KID/FID evaluation.
- Resume support ( --resume / --resume_latest ), atomic checkpoint saving, and automatic pruning.

Usage (CPU/GPU/MPS):

    python cifar10_two_class_diffusion.py --data ./data \
      --classes cat dog --batch 128 --epochs 1000 --lr 1e-4 \
      --T 1000 --guidance_scale 2.0 --p_uncond 0.2 --workers 4 \
      --save_every 2000 --plot_every 2000 --eval_every 0 \
      --resume_latest --keep_checkpoints 5

    python cifar10_two_class_diffusion.py --mode gen \
    --ckpt ./checkpoints/model_epoch2500.pt \
    --classes cat dog \
    --guidance_scale 2.0 \
    --gen_labels a b \
    --gen_counts 8 8 \
    --samples ./samples_ge

Notes:
- CIFAR-10 labels: airplane(0) automobile(1) bird(2) cat(3) deer(4) dog(5) frog(6) horse(7) ship(8) truck(9)
- Default training classes = [cat, dog]. You can choose any 2.
- Images are scaled to [-1, 1] during diffusion.
"""

import os, re, glob, time, csv, math, argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from filelock import FileLock

# plotting (optional)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -------------------------
# Helpers
# -------------------------

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """timesteps: (B,) -> (B, dim)"""
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def make_cosine_schedule(T: int, s: float = 0.008, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    # Nichol & Dhariwal cosine schedule
    t = torch.linspace(0, T, T + 1, device=device) / T
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-8, 0.999)

# -------------------------
# UNet
# -------------------------

class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        b, c, h, w = x.shape
        h_in = self.norm(x)
        h_in = h_in.view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(h_in, h_in, h_in)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        return x + attn_out

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, with_attn: bool = False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = SelfAttention2d(out_ch) if with_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_dim: int, with_attn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(out_ch + skip_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = SelfAttention2d(out_ch) if with_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 64, time_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.time_dim = time_dim
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

        self.down1 = DownBlock(base, base, time_dim, with_attn=False)      # 32->16
        self.down2 = DownBlock(base, base * 2, time_dim, with_attn=True)   # 16->8
        self.down3 = DownBlock(base * 2, base * 4, time_dim, with_attn=True) # 8->4

        self.bot1 = ResBlock(base * 4, base * 4, time_dim)
        self.bot2 = ResBlock(base * 4, base * 4, time_dim)

        self.up1 = UpBlock(base * 4, base * 2, base * 4, time_dim, with_attn=True)  # 4->8
        self.up2 = UpBlock(base * 2, base,     base * 2, time_dim, with_attn=True)  # 8->16
        self.up3 = UpBlock(base,     base,     base,     time_dim, with_attn=False) # 16->32

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        y_emb = self.class_emb(y)
        t_emb = self.time_mlp(t_emb + y_emb)

        x = self.in_conv(x)
        x1, s1 = self.down1(x, t_emb)
        x2, s2 = self.down2(x1, t_emb)
        x3, s3 = self.down3(x2, t_emb)

        x = self.bot1(x3, t_emb)
        x = self.bot2(x, t_emb)

        x = self.up1(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up3(x, s1, t_emb)

        x = F.silu(self.out_norm(x))
        x = self.out_conv(x)
        return x

# -------------------------
# Diffusion
# -------------------------

@dataclass
class DiffusionConfig:
    T: int = 1000

class Diffusion:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        betas = make_cosine_schedule(cfg.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.a_bar = alphas_cumprod
        self.sqrt_a_bar = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_a_bar = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_var = betas * (1 - alphas_cumprod.roll(1)) / (1 - alphas_cumprod)
        self.posterior_var[0] = betas[0]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_a_bar[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_a_bar[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                 guidance_scale: float = 0.0, null_class: int = 2) -> torch.Tensor:
        if guidance_scale > 0:
            y_uncond = torch.full_like(y, fill_value=null_class)
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t, t], dim=0)
            y_in = torch.cat([y_uncond, y], dim=0)
            eps = model(x_in, t_in, y_in)
            eps_uncond, eps_cond = eps.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x_t, t, y)

        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        a_bar_t = self.a_bar[t].view(-1, 1, 1, 1)

        mean = sqrt_recip_alpha_t * (x_t - beta_t * eps / torch.sqrt(1 - a_bar_t))
        var = self.posterior_var[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0). float().view(-1, 1, 1, 1)
        x_prev = mean + torch.sqrt(var) * noise * nonzero
        return x_prev.clamp(-1, 1)

    @torch.no_grad()
    def sample(self, model: nn.Module, labels: torch.Tensor, shape: Tuple[int, int, int, int],
               guidance_scale: float = 0.0, null_class: int = 2) -> torch.Tensor:
        model.eval()
        b = shape[0]
        x = torch.randn(shape, device=labels.device)
        T = self.cfg.T
        for i in reversed(range(T)):
            t = torch.full((b,), i, device=labels.device, dtype=torch.long)
            x = self.p_sample(model, x, t, labels, guidance_scale=guidance_scale, null_class=null_class)
        return x

# -------------------------
# EMA
# -------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

# -------------------------
# KID/FID Evaluator (robust)
# -------------------------

class Evaluator:
    def __init__(self, device: torch.device):
        from torchvision.models import inception_v3, Inception_V3_Weights
        # torchvision 0.15 requires aux_logits=True with pretrained weights
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.net.fc = nn.Identity()
        self.net.eval().to(device)
        self.resize = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        self.device = device

    @torch.no_grad()
    def feats(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1, 1) + 1) / 2.0
        x = self.resize(x)
        # ImageNet normalization (consistent for real/fake)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        x = (x - mean) / std
        return self.net(x.float())  # (B, 2048)

    @staticmethod
    def _cov_mean(feats: torch.Tensor):
        x = feats.double()
        mu = x.mean(dim=0)
        xc = x - mu
        cov = (xc.t() @ xc) / max(1, (x.shape[0] - 1))
        return mu, cov

    @staticmethod
    def _sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
        mat = mat.double()
        vals, vecs = torch.linalg.eigh(mat)
        vals = torch.clamp(vals, min=0)
        return (vecs @ torch.diag(vals.sqrt()) @ vecs.T)

    def fid(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        mu_r, cov_r = self._cov_mean(real)
        mu_f, cov_f = self._cov_mean(fake)
        d = mu_r.numel()
        eps = 1e-6
        I = torch.eye(d, device=mu_r.device, dtype=torch.float64)
        cov_r = cov_r + eps * I
        cov_f = cov_f + eps * I
        S = self._sqrtm_psd(cov_r)
        covmean = self._sqrtm_psd(S @ cov_f @ S)
        diff = (mu_r - mu_f).double()
        fid_val = (diff @ diff).item() + torch.trace(cov_r + cov_f - 2 * covmean).item()
        return float(max(fid_val, 1e-12))

    def kid(self, real: torch.Tensor, fake: torch.Tensor, subsets: int = 10, subset_size: int = 1000) -> float:
        # Polynomial MMD (degree=3), unbiased
        import numpy as np
        def poly_mmd2(x, y):
            d = x.shape[1]; c = 1.0; deg = 3
            k_xx = ((x @ x.t())/d + c)**deg
            k_yy = ((y @ y.t())/d + c)**deg
            k_xy = ((x @ y.t())/d + c)**deg
            m = x.shape[0]; n = y.shape[0]
            return ((k_xx.sum() - k_xx.diag().sum())/(m*(m-1))
                    + (k_yy.sum() - k_yy.diag().sum())/(n*(n-1))
                    - 2*k_xy.mean()).item()
        rng = np.random.default_rng(123)
        m = min(subset_size, real.shape[0], fake.shape[0])
        vals = []
        for _ in range(subsets):
            idx_r = torch.tensor(rng.choice(real.shape[0], m, replace=False), device=real.device)
            idx_f = torch.tensor(rng.choice(fake.shape[0], m, replace=False), device=fake.device)
            vals.append(poly_mmd2(real[idx_r], fake[idx_f]))
        return float(sum(vals)/len(vals))

    @torch.no_grad()
    def compute_metrics(self, diff: "Diffusion", model: nn.Module, real_loader: DataLoader, labels: torch.Tensor,
                        n_samples: int = 2048, batch_size: int = 128, guidance_scale: float = 2.0, null_class: int = 2,
                        kid_subsets: int = 10, kid_subset_size: int = 1000) -> dict:
        real_feats = []
        total = 0
        for x, _ in real_loader:
            x = x.to(self.device)
            real_feats.append(self.feats(x))
            total += x.size(0)
            if total >= n_samples: break
        real_feats = torch.cat(real_feats, dim=0)[:n_samples]

        fake_feats = []
        made = 0
        while made < n_samples:
            b = min(batch_size, n_samples - made)
            y = labels[made:made+b]
            xg = diff.sample(model, y, shape=(b, 3, 32, 32),
                             guidance_scale=guidance_scale, null_class=null_class)
            fake_feats.append(self.feats(xg))
            made += b
        fake_feats = torch.cat(fake_feats, dim=0)[:n_samples]

        return {'FID': self.fid(real_feats, fake_feats),
                'KID': self.kid(real_feats, fake_feats, subsets=kid_subsets, subset_size=kid_subset_size)}

# -------------------------
# Data
# -------------------------

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
NAME_TO_IDX = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}

class TwoClassCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, classes: List[str]):
        assert len(classes) == 2, "Provide exactly two class names"
        for c in classes:
            assert c in NAME_TO_IDX, f"Unknown class: {c}"
        aug = []
        if train:
            aug += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
        self.orig = datasets.CIFAR10(root=root, train=train, download=True,
                                     transform=transforms.Compose(aug + [
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                     ]))
        idx_a, idx_b = NAME_TO_IDX[classes[0]], NAME_TO_IDX[classes[1]]
        # Use raw targets (avoid calling transforms during filtering)
        targets = getattr(self.orig, 'targets', None)
        if targets is None:
            targets = self.orig.train_labels if train else self.orig.test_labels
        self.keep = [i for i, y in enumerate(targets) if y in (idx_a, idx_b)]
        self.map = {idx_a: 0, idx_b: 1}

    def __len__(self): return len(self.keep)

    def __getitem__(self, i):
        x, y = self.orig[self.keep[i]]
        return x, self.map[y]

# -------------------------
# IO helpers: atomic save & pruning
# -------------------------

def atomic_torch_save(obj, path, max_retries=3, sleep=0.5):
    """Write checkpoint atomically: save to .tmp then rename. Retries on transient errors."""
    tmp = path + ".tmp"
    err = None
    for _ in range(max_retries):
        try:
            torch.save(obj, tmp)
            try:
                with open(tmp, "rb") as f: os.fsync(f.fileno())
            except Exception:
                pass
            os.replace(tmp, path)  # atomic on POSIX
            return
        except Exception as e:
            err = e
            time.sleep(sleep)
    raise RuntimeError(f"atomic_torch_save failed for {path}: {err}")

def prune_checkpoints(checkpoints_dir: str, keep_last: int = 5):
    """Keep only the newest N epoch checkpoints by mtime."""
    files = glob.glob(os.path.join(checkpoints_dir, "model_epoch*.pt"))
    files = sorted(files, key=lambda p: os.path.getmtime(p))  # oldest -> newest
    to_delete = files[:-keep_last] if len(files) > keep_last else []
    if not to_delete:
        return
    for p in to_delete:
        try:
            os.remove(p)
        except Exception:
            pass
    print(f"Pruned {len(to_delete)} old checkpoints; kept {keep_last}.")

def find_latest_valid_checkpoint(checkpoints_dir: str, map_location="cpu") -> str:
    if not os.path.isdir(checkpoints_dir):
        return ""
    cands = sorted(glob.glob(os.path.join(checkpoints_dir, "model_epoch*.pt")), reverse=True)
    final = os.path.join(checkpoints_dir, "model_final.pt")
    if os.path.isfile(final):
        cands.append(final)
    for p in cands:
        try:
            torch.load(p, map_location=map_location)
            return p
        except Exception:
            continue
    return ""

# -------------------------
# Train / Generate
# -------------------------

def save_grid(x: torch.Tensor, path: str, nrow: int = 8):
    x = (x.clamp(-1, 1) + 1) / 2.0
    utils.save_image(x, path, nrow=nrow)

def train(args):
    torch.backends.cudnn.benchmark = True
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        (torch.device('mps') if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
         else torch.device('cpu'))
    )
    print('Using device:', device)

    classes = args.classes if len(args.classes) == 2 else ['cat','dog']
    train_ds = TwoClassCIFAR10(args.data, train=True, classes=classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    num_classes = 3  # {0,1} + null(2)
    model = UNet(in_channels=3, base=args.base, time_dim=args.time_dim, num_classes=num_classes).to(device)
    ema = EMA(model, decay=args.ema_decay)
    diff = Diffusion(DiffusionConfig(T=args.T), device)
    evaluator = Evaluator(device) if args.eval_every > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Resume
    start_epoch = 0
    global_step = 0
    resume_path = args.resume
    if args.resume_latest and not resume_path:
        resume_path = find_latest_valid_checkpoint(args.checkpoints, map_location=device)
        if resume_path:
            print(f'--resume_latest picked: {resume_path}')
    if resume_path:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=True)
        if 'ema' in ckpt: ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
        if 'opt' in ckpt:
            try: opt.load_state_dict(ckpt['opt'])
            except Exception as e: print('Warning: could not load optimizer state:', e)
        start_epoch = int(ckpt.get('epoch', 0))
        global_step = int(ckpt.get('global_step', 0))
        print(f"Loaded epoch={start_epoch}, global_step={global_step}")

    os.makedirs(args.samples, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)

    # CSV & plots
    log_csv_path = os.path.join(args.samples, args.log_csv)
    plot_png_path = os.path.join(args.samples, args.plot_png)
    metrics_csv_path = os.path.join(args.samples, args.metrics_csv)
    loss_hist, step_hist = [], []
    last_logged_step = -1

    def append_log(step, loss_val):
        nonlocal loss_hist, step_hist
        loss_hist.append(float(loss_val)); step_hist.append(int(step))
        write_header = not os.path.exists(log_csv_path)
        with open(log_csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header: w.writerow(['global_step', 'train_loss'])
            w.writerow([step, float(loss_val)])

    def append_metrics(step, metrics: dict):
        write_header = not os.path.exists(metrics_csv_path)
        with open(metrics_csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header: w.writerow(['global_step'] + list(metrics.keys()))
            w.writerow([step] + [metrics[k] for k in metrics])

    def save_plot():
        if plt is None or not loss_hist: return
        try:
            plt.figure()
            plt.plot(step_hist, loss_hist)
            plt.xlabel('global_step'); plt.ylabel('train MSE')
            plt.title('Training loss over steps')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout(); plt.savefig(plot_png_path); plt.close()
        except Exception: pass

    model.train()
    for epoch in range(start_epoch, args.epochs):
        for x0, y in train_loader:
            x0 = x0.to(device); y = y.to(device).long()

            b = x0.size(0)
            t = torch.randint(0, diff.cfg.T, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = diff.q_sample(x0, t, noise)

            # classifier-free guidance drop
            y_cf = y.clone()
            drop = torch.rand_like(y_cf.float()) < args.p_uncond
            y_cf[drop] = 2

            pred_noise = model(x_t, t, y_cf)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            global_step += 1

            if (global_step % args.log_every == 0) and (global_step != last_logged_step):
                print(f"epoch {epoch + 1}/{args.epochs} step {global_step} loss {loss.item():.4f}")
                append_log(global_step, loss.item())
                last_logged_step = global_step
                if (args.plot_every > 0) and (global_step % args.plot_every == 0):
                    save_plot()

            if global_step % args.save_every == 0:
                model_was_train = model.training
                model.eval(); ema.copy_to(model)
                with torch.no_grad():
                    labels = torch.tensor([0]*4 + [1]*4, device=device, dtype=torch.long)
                    x = diff.sample(model, labels, shape=(8, 3, 32, 32),
                                    guidance_scale=args.guidance_scale, null_class=2)
                    outp = os.path.join(args.samples, f"sample_step{global_step}.png")
                    save_grid(x, outp, nrow=4)

                    if evaluator and (args.eval_every > 0) and (global_step % args.eval_every == 0):
                        n_eval = args.eval_n
                        y_eval = torch.tensor([0,1]*((n_eval+1)//2), device=device, dtype=torch.long)[:n_eval]
                        real_loader = DataLoader(train_ds, batch_size=args.eval_batch, shuffle=False, num_workers=0)
                        metrics = evaluator.compute_metrics(
                            diff, model, real_loader, y_eval,
                            n_samples=args.eval_n, batch_size=args.eval_batch,
                            guidance_scale=args.guidance_scale, null_class=2,
                            kid_subsets=args.kid_subsets, kid_subset_size=args.kid_subset_size
                        )
                        print(f"[Eval step {global_step}] FID={metrics['FID']:.3f}  KID={metrics['KID']:.4f}")
                        append_metrics(global_step, metrics)
                if model_was_train: model.train()

        # end of epoch: EMA sample + checkpoint
        model.eval(); ema.copy_to(model)
        with torch.no_grad():
            labels = torch.tensor([0]*8 + [1]*8, device=device, dtype=torch.long)
            x = diff.sample(model, labels, shape=(16, 3, 32, 32),
                            guidance_scale=args.guidance_scale, null_class=2)
            save_grid(x, os.path.join(args.samples, f"epoch{epoch+1:03d}.png"), nrow=8)
        model.train()

        ckpt_path = os.path.join(args.checkpoints, f"model_epoch{epoch+1:03d}.pt")
        atomic_torch_save({
            'model': model.state_dict(),
            'ema': ema.shadow,
            'opt': opt.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'args': vars(args)
        }, ckpt_path)
        prune_checkpoints(args.checkpoints, keep_last=args.keep_checkpoints)
        print(f"Saved checkpoint to {ckpt_path}")

    # final save
    final_path = os.path.join(args.checkpoints, 'model_final.pt')
    atomic_torch_save({
        'model': model.state_dict(),
        'ema': ema.shadow,
        'opt': opt.state_dict(),
        'epoch': args.epochs,
        'global_step': global_step,
        'args': vars(args)
    }, final_path)

@torch.no_grad()
def generate(args):
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        (torch.device('mps') if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
         else torch.device('cpu'))
    )
    print('Using device:', device)
    ckpt = torch.load(args.ckpt, map_location=device)

    num_classes = 3
    model = UNet(in_channels=3, base=args.base, time_dim=args.time_dim, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    ema = EMA(model, decay=args.ema_decay)
    if 'ema' in ckpt:
        ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
        ema.copy_to(model)

    model.eval()
    diff = Diffusion(DiffusionConfig(T=args.T), device)
    os.makedirs(args.samples, exist_ok=True)

    labels = []
    for name, count in zip(args.gen_labels, args.gen_counts):
        idx = 0 if name == 'a' else 1
        labels += [idx] * count
    labels = torch.tensor(labels, device=device, dtype=torch.long)

    x = diff.sample(model, labels, shape=(labels.size(0), 3, 32, 32),
                    guidance_scale=args.guidance_scale, null_class=2)
    out_path = os.path.join(args.samples, "gen.png")
    save_grid(x, out_path, nrow=max(1, int(math.sqrt(labels.size(0)))))
    print("Saved samples to", out_path)

# -------------------------
# CLI
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--classes', type=str, nargs='+', default=['cat','dog'],
                   help='Two class names from: ' + ', '.join(CIFAR10_CLASSES))
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--base', type=int, default=64)
    p.add_argument('--time_dim', type=int, default=256)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--p_uncond', type=float, default=0.2, help='probability to drop label (classifier-free)')
    p.add_argument('--guidance_scale', type=float, default=2.0)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--plot_every', type=int, default=2000, help='save loss plot every N steps (0 disables)')
    p.add_argument('--log_csv', type=str, default='train_log.csv', help='CSV filename (inside samples dir)')
    p.add_argument('--plot_png', type=str, default='train_loss.png', help='PNG filename (inside samples dir)')
    p.add_argument('--metrics_csv', type=str, default='metrics_log.csv', help='CSV filename for metrics (inside samples dir)')
    p.add_argument('--samples', type=str, default='./samples')
    p.add_argument('--checkpoints', type=str, default='./checkpoints')
    p.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    p.add_argument('--ema_decay', type=float, default=0.9999)
    # evaluation
    p.add_argument('--eval_every', type=int, default=0, help='evaluate KID/FID every N steps (0 disables)')
    p.add_argument('--eval_n', type=int, default=2048, help='number of samples for eval')
    p.add_argument('--eval_batch', type=int, default=128)
    p.add_argument('--kid_subsets', type=int, default=10)
    p.add_argument('--kid_subset_size', type=int, default=1000)
    # resume / gen
    p.add_argument('--mode', type=str, choices=['train','gen'], default='train')
    p.add_argument('--ckpt', type=str, default='', help='checkpoint path for gen mode')
    p.add_argument('--gen_labels', type=str, nargs='+', default=['a','b'], help='labels for generation: a->class0, b->class1')
    p.add_argument('--gen_counts', type=int, nargs='+', default=[8,8], help='how many per label in same order as gen_labels')
    p.add_argument('--resume', type=str, default='', help='path to training checkpoint to resume from')
    p.add_argument('--resume_latest', action='store_true', help='resume from newest valid checkpoint in checkpoints dir')
    # pruning
    p.add_argument('--keep_checkpoints', type=int, default=5, help='keep only last N epoch checkpoints')
    return p

if __name__ == '__main__':
    args = build_argparser().parse_args()
    if args.mode == 'train':
        train(args)
    else:
        assert args.ckpt, "--ckpt required in gen mode"
        generate(args)
