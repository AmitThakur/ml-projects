import torch
import torch.nn.functional as F

# ---- Assumptions ----
# You have a VAE with:
#   - encode(x) -> Normal(mu, logvar) or (mu, logvar)
#   - decode(z) -> reconstructed image
# Replace `vae.encode/vae.decode` to match your API.

@torch.no_grad()
def vae_mean_latent(vae, x):
    """Return encoder mean latent z from an input image tensor x in [-1,1] or [0,1] (match your training)."""
    # Example for VAE that returns (mu, logvar)
    mu, logvar = vae.encode(x)  # shapes: [B, d], [B, d]
    return mu

def unit(v, eps=1e-9):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def tangent_edit(z, v, alpha):
    """Geodesic (norm-preserving) move on the sphere of radius ||z||."""
    # z, v: [B, d] or [d]
    if z.dim() == 1:
        z = z[None, :]
    if v.dim() == 1:
        v = v[None, :]
    z_norm = z.norm(dim=-1, keepdim=True)  # [B,1]
    # Fallback: if near origin, standard linear move
    near_origin = (z_norm < 1e-8).squeeze(-1)
    z_hat = torch.where(near_origin[:, None], z, z / (z_norm + 1e-9))
    v_perp = v - (z_hat * (z_hat * v).sum(dim=-1, keepdim=True))
    v_perp = torch.where(v_perp.norm(dim=-1, keepdim=True) < 1e-9, torch.zeros_like(v_perp), v_perp)
    v_hat = unit(v_perp)
    ca, sa = torch.cos(torch.as_tensor(alpha, device=z.device)), torch.sin(torch.as_tensor(alpha, device=z.device))
    z_prime = z_norm * (ca * z_hat + sa * v_hat)
    # Put back shape
    return z_prime[0] if z_prime.shape[0] == 1 else z_prime

# ---------- 1) Direction from two images ----------
@torch.no_grad()
def smile_direction_from_pair(vae, x_neutral, x_smile):
    """
    x_neutral, x_smile: tensors of shape [1, C, H, W], preprocessed for the VAE encoder.
    Returns a unit vector v in latent space (shape [d]).
    """
    z1 = vae_mean_latent(vae, x_neutral)  # [1, d]
    z2 = vae_mean_latent(vae, x_smile)    # [1, d]
    v = z2 - z1                            # local finite-difference direction
    v = unit(v)[0]                         # [d]
    return v

# ---------- 2) (Optional) Direction from many pairs ----------
@torch.no_grad()
def smile_direction_from_pairs(vae, neutral_list, smile_list, batch_size=8, device="cuda"):
    """
    neutral_list / smile_list: lists of tensors [C,H,W] in same order (paired).
    Returns a unit vector v (averaged direction).
    """
    dirs = []
    for i in range(0, len(neutral_list), batch_size):
        xb_n = torch.stack(neutral_list[i:i+batch_size]).to(device)  # [B,C,H,W]
        xb_s = torch.stack(smile_list[i:i+batch_size]).to(device)    # [B,C,H,W]
        z_n = vae_mean_latent(vae, xb_n)  # [B,d]
        z_s = vae_mean_latent(vae, xb_s)  # [B,d]
        v_batch = z_s - z_n               # [B,d]
        # Normalize per-pair before averaging (robust)
        v_batch = unit(v_batch)
        dirs.append(v_batch)
    v = torch.cat(dirs, dim=0).mean(dim=0)  # [d]
    v = unit(v)
    return v

# ---------- 3) Apply edit and decode ----------
@torch.no_grad()
def apply_smile_edit_and_decode(vae, x_input, v, alpha, tangent=True):
    """
    x_input: [1,C,H,W] image to edit; v: [d] unit direction; alpha: float (edit strength).
    tangent=True uses norm-preserving spherical move; else linear z' = z + alpha v.
    Returns: decoded image tensor x' and the latent z'.
    """
    z = vae_mean_latent(vae, x_input)[0]  # [d]
    if tangent:
        z_prime = tangent_edit(z, v, alpha)
    else:
        z_prime = z + alpha * v
    x_prime = vae.decode(z_prime[None, :])  # adapt to your VAE API; should return [1,C,H,W]
    return x_prime, z_prime

# ---------- 4) Example usage ----------
# vae = ...  # your loaded VAE (to(device).eval())
# x_neutral = ...  # [1,C,H,W], preprocessed
# x_smile   = ...  # [1,C,H,W], preprocessed
# v = smile_direction_from_pair(vae, x_neutral, x_smile)  # [d]
# # Edit a new face x0 with a sweep of alphas
# alphas = [-0.6, -0.3, 0.0, 0.3, 0.6]
# outs = []
# for a in alphas:
#     x_out, _ = apply_smile_edit_and_decode(vae, x0, v, alpha=a, tangent=True)
#     outs.append(x_out)  # visualize in a grid
