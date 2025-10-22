import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from tqdm import tqdm
import os
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch


names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
trainset = torchvision.datasets.CIFAR10(root=r'.', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root=r'.', train=False, download=True)
def prepare_dataset():
    trainset.transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32, 32), antialias=True),
         # For variety: Flip and crop
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0),
                                      ratio=(0.75, 1.3333333333333333),
                                      interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testset.transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32, 32), antialias=True),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



class Config:
    patch_size = 4
    hidden_size = 48
    num_hidden_layers = 4
    #  to capture different aspects of relations among tokens in the image
    num_attention_heads = 4
    intermediate_size = 4 * 48
    image_size = 32
    num_classes = 10
    num_channels = 3




def plot_dataset_sample():
    def plot_dataset_sample():
        def to_showable(img):
            # img can be PIL or Tensor; make it (H, W, C) in [0,1]
            import torch
            if isinstance(img, torch.Tensor):
                # If normalized with mean=std=0.5, undo it
                if img.ndim == 3 and img.shape[0] == 3:
                    # Denormalize if roughly in [-1, 1]
                    if img.min() < -0.5 or img.max() > 0.5:
                        img = img * 0.5 + 0.5  # undo Normalize((0.5,)*3, (0.5,)*3)
                    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                else:
                    # grayscale tensor or other shapes
                    img = img.squeeze().cpu().numpy()
            else:
                # PIL -> numpy in HWC
                img = np.array(img)
                # If it came as uint8 0-255, scale to 0-1 for consistency (optional)
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
            return img

        plt.figure(figsize=(12, 6), dpi=300)
        for i in range(3):
            for j in range(6):
                plt.subplot(3, 6, 6 * i + j + 1)
                img, label = trainset[6 * i + j]
                plt.imshow(to_showable(img))
                plt.axis('off')
                plt.title(names[label], fontsize=12)
        plt.subplots_adjust(hspace=0.20)
        plt.savefig('trainset_samples.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()


def plot_loss_curve(train_losses, test_losses, outpath="files/loss_curve.png"):
    """
    Save a simple loss curve figure for train/test losses over epochs.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 4), dpi=200)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ViT Training / Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


class GELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * \
                                               (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv2d(config.num_channels,
                                    config.hidden_size,
                                    kernel_size=config.patch_size,
                                    stride=config.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1,
                                                  config.hidden_size))
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x


class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size)
            self.heads.append(head)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output,
        _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _,
            attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output,
                                                      output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0, :])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0, std=0.02, ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0, std=0.02, ).to(module.cls_token.dtype)

class EarlyStop:
    def __init__(self, patience=3):
        self.patience = patience
        self.steps = 0
        self.min_loss = float('inf')
    def stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.steps = 0
            to_save = True
        elif loss >= self.min_loss:
            self.steps += 1
            to_save = False
        if self.steps >= self.patience:
            to_stop = True
        else:
            to_stop = False
        return to_save, to_stop

def train_batch(batch, train_loader):
    batch = [t.to(device) for t in batch]
    images, labels = batch
    with torch.amp.autocast(str(device)):
        loss = loss_fn(model(images)[0], labels)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item() * len(images)/len(train_loader.dataset)

if __name__ == '__main__':

    train_loader, test_loader = prepare_dataset()
    plot_dataset_sample()
    config = Config()
    device = get_device()
    patch_embed = PatchEmbeddings(config)
    # create a hypothetical image
    img = torch.randn((1, 3, 32, 32))
    # pass a batch of one image through the class
    out = patch_embed(img)
    # print out the shape of the output
    print(out.shape)

    embed = Embeddings(config)
    # create a hypothetical image
    img = torch.randn((1, 3, 32, 32))
    # pass a batch of one image through the class
    out = embed(img)
    # the shapes of the output
    print(out.shape)
    # the shapes of the positional encoding
    print(embed.position_embeddings.shape)
    model = ViTForClassfication(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    stopper = EarlyStop()
    os.makedirs("files", exist_ok=True)
    train_losses_history = []
    test_losses_history = []

    for i in range(100):
        print(f'Epoch {i + 1}')
        model.train()
        trainL, testL = 0, 0
        for batch in tqdm(train_loader):
            loss = train_batch(batch, train_loader)
            trainL += loss
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = [t.to(device) for t in batch]
                images, labels = batch
                logits, _ = model(images)
                loss = loss_fn(logits, labels)
                testL += loss.item() * len(images) / len(test_loader.dataset)
        print(f'Train and test losses: {trainL:.4f}, {testL:.4f}')
        # Record history and update the loss curve plot
        train_losses_history.append(trainL)
        test_losses_history.append(testL)
        plot_loss_curve(train_losses_history, test_losses_history, outpath="files/loss_curve.png")

        to_save, to_stop = stopper.stop(testL)
        if to_save == True:
            torch.save(model.state_dict(), "files/ViT.pth")
        if to_stop == True:
            break

        # model.load_state_dict(torch.load('files/ViT.pth'))
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            batch = [t.to(device) for t in batch]
            images, labels = batch
            logits, attention_maps = model(images, output_attentions=True)
            predictions = torch.argmax(logits, dim=1)

        print(predictions)
        print([names[i] for i in predictions.tolist()])

        with torch.no_grad():
            attention_maps = torch.cat(attention_maps, dim=1)
            print(f"attention map shape: {attention_maps.shape}")
            attention_maps = attention_maps[:, :, 0, 1:]
            print(f"attention map shape: {attention_maps.shape}")
            attention_maps = attention_maps.mean(dim=1)
            print(f"attention map shape: {attention_maps.shape}")
            num_patches = attention_maps.size(-1)
            size = int(math.sqrt(num_patches))
            attention_maps = attention_maps.view(-1, size, size)
            print(f"attention map shape: {attention_maps.shape}")
            attention_maps = attention_maps.unsqueeze(1)
            attention_maps = F.interpolate(attention_maps, size=(32, 32),
                                           mode='bilinear', align_corners=False)
            attention_maps = attention_maps.squeeze(1)  # D
            print(f"attention map shape: {attention_maps.shape}")

        fig = plt.figure(figsize=(8, 8), dpi=100)
        for i in range(16):
            ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
            ax.imshow(attention_maps[i].cpu(), alpha=0.5, cmap='jet')
        plt.tight_layout()
        plt.savefig('files/attention_map.png')
        plt.close()

        fig = plt.figure(figsize=(8, 5), dpi=100)
        mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))],
                              axis=1)
        for i in range(16):
            ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
            img = np.concatenate((images[i].cpu(), images[i].cpu()),
                                 axis=-1)
            ax.imshow(img.transpose(1, 2, 0) / 2 + 0.5)
            extended_attention_map = np.concatenate((np.zeros((32, 32)),
                                                     attention_maps[i].cpu()),
                                                    axis=1)
            extended_attention_map = np.ma.masked_where(mask == 1,
                                                        extended_attention_map)
            ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
            gt = names[labels[i]]
            pred = names[predictions[i]]
            ax.set_title(f"Actual: {gt} / Pred: {pred}",
                         color=("green" if gt == pred else "red"),
                         fontsize=10)
        plt.tight_layout()
        plt.savefig('files/prediction.png')
        plt.close()



