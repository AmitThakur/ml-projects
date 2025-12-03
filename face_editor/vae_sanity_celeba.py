import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from PIL import Image
from diffusers import AutoencoderKL


class LocalCelebADataset(Dataset):
    """
    Minimal CelebA dataset using existing files:
        root/
          img_align_celeba/
          list_attr_celeba.txt
          list_eval_partition.txt
    No download / integrity check.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform

        img_dir = os.path.join(self.root, "img_align_celeba")
        attr_path = os.path.join(self.root, "list_attr_celeba.txt")
        part_path = os.path.join(self.root, "list_eval_partition.txt")

        if not os.path.isdir(img_dir):
            raise RuntimeError(f"img_align_celeba not found at {img_dir}")
        if not os.path.isfile(attr_path):
            raise RuntimeError(f"list_attr_celeba.txt not found at {attr_path}")
        if not os.path.isfile(part_path):
            raise RuntimeError(f"list_eval_partition.txt not found at {part_path}")

        # partitions (0=train, 1=val, 2=test)
        part_dict = {}
        with open(part_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, pid = line.split()
                part_dict[fname] = int(pid)

        # attributes
        with open(attr_path, "r") as f:
            lines = f.readlines()
        attr_names = lines[1].split()

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
        self.attr_names = attr_names
        self.samples = data

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname, attrs = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, attrs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_root = "/home/jovyan/data/celeba"  # <-- change if needed
    image_size = 256

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                      # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),      # [-1,1]
    ])

    ds = LocalCelebADataset(root=data_root, split="train", transform=transform)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    imgs, attrs = next(iter(dl))
    imgs = imgs.to(device)

    with torch.no_grad():
        enc = vae.encode(imgs)
        # deterministic latents: use mean, scale by 0.18215
        latents = enc.latent_dist.mean * 0.18215
        rec = vae.decode(latents / 0.18215).sample

    os.makedirs("./vae_sanity_out", exist_ok=True)
    vutils.save_image(
        imgs,
        "./vae_sanity_out/orig.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )
    vutils.save_image(
        rec,
        "./vae_sanity_out/recon.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )

    print("Saved ./vae_sanity_out/orig.png and recon.png")


if __name__ == "__main__":
    main()
