# transformer.py
"""
python transformer.py --mode train --epochs 50 --ckpt files/de2en.pth
python transformer.py --mode infer --ckpt files/de2en.pth \
  --examples "Ein Mann in einem blauen Hemd spielt Gitarre.,Zwei Kinder laufen am Strand."

"""
import argparse
import csv
import math
import os
import sys
import tarfile
from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from torch import nn
from tqdm import tqdm



# ---------------------------
# Constants & Device
# ---------------------------
PAD = 0
UNK = 1
BATCH_SIZE = 128


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ---------------------------
# Data utilities
# ---------------------------
def download_dataset():
    url = (
        "https://raw.githubusercontent.com/neychev/"
        "small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    )
    os.makedirs("files", exist_ok=True)
    tgz = "files/training.tar.gz"
    if not os.path.exists(tgz):
        dnload = requests.get(url)
        with open(tgz, "wb") as f:
            f.write(dnload.content)
    with tarfile.open(tgz) as train:
        train.extractall("files")


def prepare_dictionary(tokens, max_vocab=50000):
    word_count = Counter()
    for sentence in tokens:
        for word in sentence:
            word_count[word] += 1
    frequency = word_count.most_common(max_vocab)
    # token -> id
    word_dict = {w[0]: idx + 2 for idx, w in enumerate(frequency)}
    word_dict["PAD"] = PAD
    word_dict["UNK"] = UNK
    # id -> token
    idx_dict = {v: k for k, v in word_dict.items()}
    return idx_dict, word_dict


def seq_padding(X, padding=PAD):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array(
        [x + [padding] * (ML - len(x)) if len(x) < ML else x for x in X],
        dtype=np.int64,
    )


def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def make_std_mask(tgt: torch.Tensor, pad: int):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    return tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)


class Batch:
    """
    Holds a padded mini-batch and its masks.
    """
    def __init__(self, src, trg=None, pad=PAD):
        src = torch.from_numpy(src).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = torch.from_numpy(trg).to(DEVICE).long()
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()


def prepare_dataset():
    """
    Returns:
      - batches: List[Batch] (shuffled, length-bucketed once)
      - src_vocab: int (DE vocab size)
      - tgt_vocab: int (EN vocab size)
      - de_word_dict: dict[str->int]  (DE tok->id)
      - en_word_dict: dict[str->int]  (EN tok->id)
      - en_idx_dict: dict[int->str]   (EN id->tok)
      - de_tokenizer: spaCy tokenizer for DE
    """
    with open("files/train.de", "rb") as fb:
        train_de = [i.decode("utf-8").strip() for i in fb.readlines()]
    with open("files/train.en", "rb") as fb:
        train_en = [i.decode("utf-8").strip() for i in fb.readlines()]

    import spacy
    import subprocess

    def load_spacy_model(name):
        try:
            return spacy.load(name)
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", name], check=True)
            return spacy.load(name)

    de_tokenizer = load_spacy_model("de_core_news_sm")
    en_tokenizer = load_spacy_model("en_core_web_sm")

    # Tokenize full corpora with BOS/EOS
    en_tokens = [["BOS"] + [tok.text for tok in en_tokenizer.tokenizer(x)] + ["EOS"] for x in train_en]
    de_tokens = [["BOS"] + [tok.text for tok in de_tokenizer.tokenizer(x)] + ["EOS"] for x in train_de]

    # Build vocabularies
    en_idx_dict, en_word_dict = prepare_dictionary(en_tokens)
    de_idx_dict, de_word_dict = prepare_dictionary(de_tokens)

    # Numericalize corpora
    out_en_ids = [[en_word_dict.get(w, UNK) for w in s] for s in en_tokens]
    out_de_ids = [[de_word_dict.get(w, UNK) for w in s] for s in de_tokens]

    # Sort by source length (keeps alignment)
    sorted_ids = sorted(range(len(out_de_ids)), key=lambda x: len(out_de_ids[x]))
    out_de_ids = [out_de_ids[x] for x in sorted_ids]
    out_en_ids = [out_en_ids[x] for x in sorted_ids]

    # Bucket once for training convenience
    idx_list = np.arange(0, len(out_de_ids), BATCH_SIZE)
    np.random.shuffle(idx_list)

    batches = []
    for idx in idx_list:
        end = min(len(out_de_ids), idx + BATCH_SIZE)
        batch_de = [out_de_ids[i] for i in range(idx, end)]
        batch_en = [out_en_ids[i] for i in range(idx, end)]
        batch_de = seq_padding(batch_de, padding=PAD)
        batch_en = seq_padding(batch_en, padding=PAD)
        batches.append(Batch(batch_de, batch_en))

    return (
        batches,
        len(de_word_dict),
        len(en_word_dict),
        de_word_dict,
        en_word_dict,
        en_idx_dict,
        de_tokenizer,
    )

# ---------------------------
# Model components
# ---------------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0.0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))
        pe_pos = position * div_term
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # include a nonlinearity (ReLU) for better capacity
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_z = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x_z + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([deepcopy(SublayerConnection(size, dropout)) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([deepcopy(SublayerConnection(size, dropout)) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    Projects decoder outputs to vocabulary logits, then log-softmax.
    """
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def create_model(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    pos = PositionalEncoding(d_model, dropout).to(DEVICE)
    model = Transformer(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), deepcopy(pos)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), deepcopy(pos)),
        Generator(d_model, tgt_vocab),
    ).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ---------------------------
# Loss & Optimizer
# ---------------------------
class LabelSmoothing(nn.Module):
    """
    KLDivLoss with smoothed targets. Set smoothing=0.0 to recover NLL.
    """
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        # x: (N, V) log-probs, target: (N,)
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())


class NoamOpt:
    """
    Optimizer wrapper implementing the learning rate schedule from Attention Is All You Need.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0.0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


# ---------------------------
# Plotting
# ---------------------------
def plot_loss(losses, save_path="files/train_loss.png"):
    plt.figure(figsize=(7, 4), dpi=120)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Avg NLL (per token)")
    plt.title("Transformer Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------
# Inference helpers
# ---------------------------
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, bos_id, eos_id):
    """
    Greedy decoding for a single example.
    """
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[bos_id]], device=DEVICE, dtype=torch.long)
    for _ in range(max_len - 1):
        tgt_mask = make_std_mask(ys, pad=PAD).to(DEVICE)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        log_prob = model.generator(out[:, -1])  # (1, V)
        next_word = torch.argmax(log_prob, dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
        if next_word == eos_id:
            break
    return ys.squeeze(0).tolist()


def detok_en(tokens):
    toks = [t for t in tokens if t not in ("BOS", "EOS", "PAD")]
    s = " ".join(toks)
    for p in [" .", " ,", " !", " ?", " '", " )", " ;", " :", " %"]:
        s = s.replace(p, p[1:])
    s = s.replace("( ", "(")
    return s


def encode_de(sentence_str, de_tokenizer, de_word_dict):
    toks = ["BOS"] + [tok.text for tok in de_tokenizer.tokenizer(sentence_str)] + ["EOS"]
    ids = [de_word_dict.get(t, UNK) for t in toks]
    src = torch.tensor([ids], device=DEVICE, dtype=torch.long)  # (1, S)
    src_mask = (src != PAD).unsqueeze(-2)  # (1,1,S)
    return src, src_mask


def translate_sentence(sentence_de, model, de_tokenizer, de_word_dict, en_word_dict, en_idx_dict, max_len=60):
    src, src_mask = encode_de(sentence_de, de_tokenizer, de_word_dict)
    bos_id = en_word_dict["BOS"]
    eos_id = en_word_dict["EOS"]
    ids = greedy_decode(model, src, src_mask, max_len, bos_id, eos_id)
    toks = [en_idx_dict.get(i, "UNK") for i in ids]
    return detok_en(toks)


# ---------------------------
# Train / Infer runners
# ---------------------------
def run_training(epochs: int, ckpt_path: str):
    os.makedirs("files", exist_ok=True)
    download_dataset()
    (
        batches,
        src_vocab,
        tgt_vocab,
        de_word_dict,
        en_word_dict,
        en_idx_dict,
        de_tokenizer,
    ) = prepare_dataset()

    model = create_model(src_vocab, tgt_vocab, N=6, d_model=256, d_ff=1024, h=8, dropout=0.1)
    optimizer = NoamOpt(
        256,
        1,
        2000,
        torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9),
    )
    # Turn smoothing on/off as desired:
    criterion = LabelSmoothing(tgt_vocab, padding_idx=PAD, smoothing=0.0)
    loss_func = SimpleLossCompute(model.generator, criterion, optimizer)

    epoch_losses = []
    csv_path = "files/train_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss_per_token", "learning_rate"])

    for epoch in range(epochs):
        model.train()
        tloss = 0
        tokens = 0
        for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_func(out, batch.trg_y, batch.ntokens)
            tloss += loss
            tokens += batch.ntokens
        avg_loss = (tloss / tokens).item() if hasattr(tloss, "item") else float(tloss / tokens)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1:02d}: avg loss/token = {avg_loss:.4f}, lr = {optimizer._rate:.6e}")
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, optimizer._rate])

    torch.save(model.state_dict(), ckpt_path)
    plot_loss(epoch_losses, save_path="files/train_loss.png")
    print("Saved:", csv_path, "and files/train_loss.png")
    print(f"Checkpoint saved to {ckpt_path}")


def run_inference(ckpt_path: str, examples: list[str], max_len: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.makedirs("files", exist_ok=True)
    download_dataset()
    (
        _batches,
        src_vocab,
        tgt_vocab,
        de_word_dict,
        en_word_dict,
        en_idx_dict,
        de_tokenizer,
    ) = prepare_dataset()

    model = create_model(src_vocab, tgt_vocab, N=6, d_model=256, d_ff=1024, h=8, dropout=0.1)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("\nTranslations (greedy):")
    for s in examples:
        en = translate_sentence(s, model, de_tokenizer, de_word_dict, en_word_dict, en_idx_dict, max_len=max_len)
        print(f"DE: {s}\nEN: {en}\n")


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or run inference for a simple DE→EN Transformer.")
    parser.add_argument(
        "--mode",
        choices=["train", "infer"],
        required=True,
        help="Choose 'train' to train the model or 'infer' to run translation.",
    )
    parser.add_argument("--ckpt", default="files/de2en.pth", help="Path to save/load the model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (used in --mode train).")
    parser.add_argument(
        "--examples",
        type=str,
        default=(
            "Ein Mann in einem blauen Hemd spielt Gitarre.,"
            "Zwei Kinder laufen am Strand.,"
            "Eine Frau sitzt auf einer Bank im Park."
        ),
        help="Comma-separated German sentences for inference (used in --mode infer).",
    )
    parser.add_argument("--max_len", type=int, default=60, help="Max decoding length for inference.")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(epochs=args.epochs, ckpt_path=args.ckpt)
    else:
        examples = [s.strip() for s in args.examples.split(",") if s.strip()]
        run_inference(ckpt_path=args.ckpt, examples=examples, max_len=args.max_len)


if __name__ == "__main__":
    main()
