# make_celeba_smile_pairs.py (patched, robust + faster)
"""
python make_celeba_smile_pairs.py --num_pairs 100
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple
import random

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CelebA

def load_celeba(root: str, split: str) -> CelebA:
    """
    split in {'train', 'valid', 'test', 'all'}
    """
    ds = CelebA(
        root=root,
        split=split,
        target_type=['attr', 'identity'],
        download=False
    )
    return ds

def get_smile_col(ds: CelebA) -> torch.Tensor:
    """
    Returns a 1D tensor of length N with boolean smile flags.
    Works for both encodings: {-1,+1} or {0,1}.
    """
    # Attrs: shape [N, 40]
    attrs = ds.attr
    assert attrs is not None, "CelebA attrs not found. Check torchvision version/data."
    smile_idx = ds.attr_names.index("Smiling")

    col = attrs[:, smile_idx]  # shape [N]
    # Normalize to boolean smiling
    if col.min().item() >= 0:  # 0/1 encoding
        smiling = col.bool()
    else:  # -1/+1 encoding
        smiling = col > 0
    return smiling

def get_identity(ds: CelebA) -> torch.Tensor:
    """
    Returns identity as a 1D long tensor [N].
    """
    ident = ds.identity
    if ident is None:
        # Fallback: build via per-item fetch (slow—but should almost never be needed)
        print("Warning: ds.identity missing; falling back to slow per-item read.")
        id_list = []
        for i in range(len(ds)):
            _, (_, identity) = ds[i]
            id_list.append(int(identity))
        ident = torch.tensor(id_list, dtype=torch.long)
    else:
        ident = ident.view(-1).to(torch.long)
    return ident

def compute_eligible_identities(smiling: torch.Tensor, ident: torch.Tensor) -> List[int]:
    """
    Returns identity ids that have at least one smiling and one non-smiling image.
    """
    neutral = ~smiling
    # For each identity, check if both exist
    # We’ll count per identity with boolean reductions.
    ids = ident.unique().tolist()

    elig = []
    # Pre-index masks for speed
    for pid in ids:
        m = (ident == pid)
        if m.any():
            has_smile = (smiling & m).any().item()
            has_neutral = (neutral & m).any().item()
            if has_smile and has_neutral:
                elig.append(pid)
    return elig

def pick_pairs(elig_ids: List[int], smiling: torch.Tensor, ident: torch.Tensor,
               num_pairs: int, seed: int) -> List[Tuple[int, int, int]]:
    """
    Returns list of tuples: (identity, neutral_idx, smile_idx)
    """
    rng = random.Random(seed)
    if len(elig_ids) < num_pairs:
        raise RuntimeError(
            f"Not enough eligible identities with both neutral & smiling images. "
            f"Found {len(elig_ids)}, requested {num_pairs}."
        )
    rng.shuffle(elig_ids)
    chosen = elig_ids[:num_pairs]

    pairs = []
    neutral = ~smiling
    # Build index lists once for speed
    N = ident.numel()
    for pid in chosen:
        # indices for this identity
        idxs = torch.arange(N)[ident == pid]
        neu_idxs = idxs[neutral[idxs]]
        smi_idxs = idxs[smiling[idxs]]
        # choose one from each
        n_idx = int(neu_idxs[rng.randrange(len(neu_idxs))].item())
        s_idx = int(smi_idxs[rng.randrange(len(smi_idxs))].item())
        pairs.append((int(pid), n_idx, s_idx))
    return pairs

def safe_filename(ds: CelebA, idx: int) -> str:
    if hasattr(ds, "filename") and isinstance(ds.filename, list) and len(ds.filename) == len(ds):
        return ds.filename[idx]
    return ""

def save_pairs(ds: CelebA, pairs, outdir: Path, resize: int = None, jpeg_quality: int = 95) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (pid, neutral_idx, smile_idx) in enumerate(tqdm(pairs, desc="Saving pairs"), start=1):
        # Load images via __getitem__
        neutral_img, _ = ds[neutral_idx]
        smile_img, _ = ds[smile_idx]
        if resize is not None:
            neutral_img = neutral_img.resize((resize, resize), Image.BICUBIC)
            smile_img = smile_img.resize((resize, resize), Image.BICUBIC)

        pair_id = f"{k:04d}"
        fn_neu = f"{pair_id}_neutral.jpg"
        fn_smi = f"{pair_id}_smile.jpg"

        neutral_img.save(outdir / fn_neu, format="JPEG", quality=jpeg_quality, subsampling=1)
        smile_img.save(outdir / fn_smi, format="JPEG", quality=jpeg_quality, subsampling=1)

        rows.append({
            "pair_id": pair_id,
            "identity": int(pid),
            "neutral_index": neutral_idx,
            "smile_index": smile_idx,
            "neutral_file": fn_neu,
            "smile_file": fn_smi,
            "orig_neutral": safe_filename(ds, neutral_idx),
            "orig_smile": safe_filename(ds, smile_idx),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pairs.csv", index=False)
    return df

def main():
    ap = argparse.ArgumentParser("Build neutral/smile identity-matched pairs from CelebA")
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--out", type=str, default="./pairs")
    ap.add_argument("--num_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="all", choices=["train", "valid", "test", "all"])
    ap.add_argument("--resize", type=int, default=None, help="Optional square resize (e.g., 256)")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CelebA split='{args.split}' to {args.root} ...")
    ds = load_celeba(args.root, args.split)

    smiling = get_smile_col(ds)         # boolean [N]
    ident = get_identity(ds)            # long [N]
    N = len(ds)

    # Diagnostics
    n_smile = int(smiling.sum().item())
    n_neutral = int((~smiling).sum().item())
    uniq_ids = ident.unique().numel()
    print(f"Total images: {N}")
    print(f"Smiling: {n_smile} | Neutral: {n_neutral}")
    print(f"Unique identities in split: {int(uniq_ids)}")

    elig_ids = compute_eligible_identities(smiling, ident)
    print(f"Eligible identities (have both neutral & smiling): {len(elig_ids)}")

    pairs = pick_pairs(elig_ids, smiling, ident, args.num_pairs, args.seed)
    print(f"Saving {len(pairs)} pairs to: {outdir}")
    df = save_pairs(ds, pairs, outdir, resize=args.resize)
    print(f"Done. CSV manifest at: {outdir / 'pairs.csv'}")
    print(df.head(min(5, len(df))))

if __name__ == "__main__":
    main()
