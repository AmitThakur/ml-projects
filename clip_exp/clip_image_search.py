# python clip_image_search.py train
# python clip_image_search.py inference --prompt "A dog in a park"

import PIL
from matplotlib import pyplot as plt
import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import albumentations as A

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import pickle
import argparse

best_loss = 100.0

class CFG:
    image_path = r"files/Images"
    captions_path = r"files"
    batch_size = 32
    head_lr = 1e-3
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    pretrained = True  # for both image encoder and text encoder
    trainable = False  # for both image encoder and text encoder
    temperature = 1.0
    # image size
    size = 224
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1


# cfg=CFG()

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer,
                 transforms):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True,
            max_length=CFG.max_length)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx])
                for key, values in self.encoded_captions.items()}
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)


def get_transforms():
    return A.Compose([
        A.Resize(CFG.size, CFG.size, always_apply=True),
        A.Normalize(max_pixel_value=255.0,
                    always_apply=True), ])


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.model_name,
                 pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained,
                                       num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model,
                 pretrained=CFG.pretrained,
                 trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


# To make the dimensions of the text embedding and image embedding the same
# (CFG.projection_dim), we use ProjectionHead.
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim,
                 projection_dim=CFG.projection_dim,
                 dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim,
                                    projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature,
                 image_embedding=CFG.image_embedding,
                 text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def prepare_data():
    df = pd.read_csv(r'files/captions.txt', delimiter=",")
    print(df.head(n=12))
    visualize_data(df)

    # split the dataset into train and validation subsets
    image_ids = np.arange(0, len(df))
    np.random.seed(42)
    valid_ids = np.random.choice(image_ids,
                                 size=int(0.2 * len(df)),
                                 replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]

    train = df[df.index.isin(train_ids)].reset_index(drop=True)
    valid = df[df.index.isin(valid_ids)].reset_index(drop=True)

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    transforms = get_transforms()

    # Each batch is a Python dictionary with four key-value pairs.
    # The keys (within CLIPDataset) are 'input_ids', 'attention_mask', 'image',
    # and 'caption'.
    # - The input_ids for a caption is a sequence of integers.
    # The sequence is added with 0 at the end to make sure that all
    # sequences in the batch have the same length.
    # - The 'attention_mask' vector mask out the paddings at the end so
    # the model pays attention to the indexes corresponding to the captions,
    # not to the 0s at the end of the sequence.
    # - Each image is now represented by a tensor with a size of
    # (3, CFG.size, CFG.size)

    train_set = CLIPDataset(train["image"].values,
                            train["caption"].values, tokenizer=tokenizer,
                            transforms=transforms)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=CFG.batch_size,
                                               shuffle=True)
    val_set = CLIPDataset(valid["image"].values,
                          valid["caption"].values, tokenizer=tokenizer,
                          transforms=transforms)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=CFG.batch_size,
                                             shuffle=False)
    return train_loader, val_loader, valid, tokenizer


def visualize_data(df):
    img_folder = r"files/Images"

    with os.scandir(img_folder) as fb:
        files = [f.name for f in fb]
    start = 100
    imgs = files[start:start + 10]
    dfi = df[df["image"].isin(imgs)].copy()
    dfi["length"] = dfi["caption"].str.len()
    dfi = dfi.sort_values(['image', "length"])
    dfi = dfi.groupby("image").first()

    plt.figure(dpi=200, figsize=(15, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        img = f"{img_folder}/{dfi.index[i]}"
        nparray = PIL.Image.open(img)
        plt.imshow(nparray)
        plt.title(f"{dfi.iloc[i]['caption']}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('files/visualize.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate(model, val_loader):
    global best_loss
    model.eval()
    losses = []
    with torch.no_grad():
        tqdm_object = tqdm(val_loader, total=len(val_loader))
        for batch in tqdm_object:
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)
            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            tqdm_object.set_description(f"valid_loss={avg_loss:.5f}")

    # save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs("files", exist_ok=True)
        torch.save(model.state_dict(), "files/best.pth")
        print("Saved Best Model!")

    return avg_loss


def train(model, train_loader, val_loader):

    num_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_trainable}")
    non_trainable = sum([p.numel() for p in model.parameters() if not p.requires_grad])
    print(f"Number of untrainable parameters: {non_trainable}")

    # Only optimize projection heads
    params = [
        {
            "params": itertools.chain(
                model.image_projection.parameters(),
                model.text_projection.parameters()
            ),
            "lr": CFG.head_lr,
            "weight_decay": CFG.weight_decay
        }
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    scaler = torch.amp.GradScaler("cuda")


    train_losses_per_epoch = []
    val_losses_per_epoch = []

    EPOCHS = 10
    os.makedirs("files", exist_ok=True)

    def save_loss_plot():
        # Plot Train vs Val loss
        plt.figure(figsize=(8, 5), dpi=120)
        plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, label="Train Loss")
        plt.plot(range(1, len(val_losses_per_epoch) + 1), val_losses_per_epoch, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CLIP Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("files/loss_curve.png")
        plt.close()

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}")
        model.train()
        losses = []
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        avg_loss = 0.0

        for batch in tqdm_object:
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}

            with torch.amp.autocast("cuda"):
                loss = model(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            tqdm_object.set_description(f"loss={avg_loss:.5f}")

        # record epoch train loss
        train_losses_per_epoch.append(avg_loss)

        # evaluate returns val avg
        val_avg = evaluate(model, val_loader)
        val_losses_per_epoch.append(val_avg)

        # scheduler on val loss (typical)
        lr_scheduler.step(val_avg)

        # save an updated plot after each epoch
        save_loss_plot()

    # also save once more at the end (no-op if unchanged)
    save_loss_plot()
    print("Saved training curve to files/loss_curve.png")


def match(model, prompt, tokenizer, image_embeddings, valid, k=1):
    encoded = tokenizer([prompt])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    dot_similarity = text_embeddings @ image_embeddings.T
    values, idx = torch.topk(dot_similarity.squeeze(0), k)

    img = valid['image'].values[idx.item()]
    caption = valid['caption'].values[idx.item()]
    return img, caption

def inference_prep(model, val_loader):
    # obtain the image embeddings of all images in the validation set
    # and use them as potential candidates for a match for any text prompt
    image_embeds = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image_features = model.image_encoder(
                batch["image"].to(CFG.device))
            image_embeds.append(
                model.image_projection(image_features))
    image_embeddings = torch.cat(image_embeds)

    # save the image embeddings on the computer so that we
    # don't need to create image embeddings again later.
    with open("files/image_embeds.p", "wb") as f:
        pickle.dump(image_embeddings, f)


def run_inference(prompt, tokenizer, valid, model):
    sd = torch.load(r"files/best.pth")
    model.load_state_dict(sd)
    model.eval()

    with open("files/image_embeds.p", "rb") as f:
        image_embeddings = pickle.load(f)
        file, cap = match(model, prompt,
                          tokenizer=tokenizer,
                          image_embeddings=image_embeddings,
                          valid=valid)
        plt.imshow(PIL.Image.open(rf"files/Images/{file}"))
        plt.title(f"Prompt: {prompt}\nOriginal caption: {cap}")
        plt.axis("off")
        plt.savefig("files/inference.png")
        plt.close()
        print('File saved as files/inference.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or run inference with the Transformer model.")
    parser.add_argument(
        "mode",
        choices=["train", "inference"],
        help="Mode of operation: 'train' to train the model or 'inference' to generate text."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for inference (required if mode='inference')."
    )

    args = parser.parse_args()
    train_loader, val_loader, valid, tokenizer = prepare_data()
    model = CLIPModel().to(CFG.device)

    if args.mode == "train":
        train(model, train_loader, val_loader)
        inference_prep(model, val_loader)
    elif args.mode == "inference":
        if args.prompt is None:
            raise ValueError("You must provide --prompt when running in inference mode.")
        run_inference(args.prompt, tokenizer, valid, model)


