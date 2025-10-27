import json
from collections import Counter
import json, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch.distributions import Categorical
import torchvision
import matplotlib.pyplot as plt


def center_crop(img):
    width, height = img.size
    new = min(width, height)
    left = (width - new) / 2
    top = (height - new) / 2
    right = (width + new) / 2
    bottom = (height + new) / 2
    im = img.crop((left, top, right, bottom))
    return im


class FlickrD(Dataset):
    def __init__(self, images, captions, word2idx):
        self.images = images
        self.captions = captions
        self.word2idx = word2idx
        self._max_len = 50
        self.image_size = 128
        self._image_transform = self._construct_image_transform(self.image_size)
        self._data = self._create_input_label_mappings()
        self._dataset_size = len(self._data)
        self._start_idx = 1
        self._end_idx = 2
        self._pad_idx = 0
        self._UNK_idx = 3
        self._START_token = "<start>"
        self._END_token = "<end>"
        self._PAD_token = "<pad>"
        self._UNK_token = "<unk>"

    def _construct_image_transform(self, image_size):
        # ImageNet normalization statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            normalize, ])
        return preprocessing

    def _load_and_process_images(self):
        # Load images
        images_raw = [center_crop(Image.open(path)).resize((self.image_size,
                                                            self.image_size)) for path in self.images]
        # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
        image_tensors = [self._image_transform(img) for img in images_raw]
        images_processed = {img_name: img_tensor for img_name, img_tensor
                            in zip(self.images, image_tensors)}
        return images_processed

    def _group_captions(self):
        grouped_captions = {self.images[i]: self.captions[i]
                            for i in range(len(self.images))}
        return grouped_captions

    def _create_input_label_mappings(self):
        processed_data = []
        for img, caps in self._group_captions().items():
            for cap in caps:
                pair = (img, cap)
                processed_data.append(pair)
        return processed_data

    def _load_and_prepare_image(self, image_name):
        img_pil = center_crop(Image.open(image_name)).resize((self.image_size,
                                                              self.image_size))
        image_tensor = self._image_transform(img_pil)
        return image_tensor

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        # Extract the caption data
        image_id, tokens = self._data[index]
        # Load and preprocess image
        image_tensor = self._load_and_prepare_image(image_id)
        # Pad the token and label sequences
        tokens = tokens[:self._max_len]
        tokens = [token.strip().lower() for token in tokens]
        tokens = [self._START_token] + tokens + [self._END_token]
        # Extract input and target output
        input_tokens = tokens[:-1].copy()
        tgt_tokens = tokens[1:].copy()

        # Number of words in the input token
        sample_size = len(input_tokens)
        padding_size = self._max_len - sample_size

        if padding_size > 0:
            padding_vec = [self._PAD_token for _ in range(padding_size)]
            input_tokens += padding_vec.copy()
            tgt_tokens += padding_vec.copy()

        # Apply the vocabulary mapping to the input tokens
        input_tokens = [self.word2idx.get(token, self._UNK_idx)
                        for token in input_tokens]
        tgt_tokens = [self.word2idx.get(token, self._UNK_idx)
                      for token in tgt_tokens]

        input_tokens = torch.Tensor(input_tokens).long()
        tgt_tokens = torch.Tensor(tgt_tokens).long()

        # Index from which to extract the model prediction
        # Define the padding masks
        attn_mask = torch.zeros([self._max_len, ])
        attn_mask[:sample_size] = 1.0
        attn_mask = attn_mask.bool()

        return image_tensor, input_tokens, tgt_tokens, attn_mask


# Convert an image to patches.
def extract_patches(image_tensor, patch_size=16):
    # Get the dimensions of the image tensor
    bs, c, h, w = image_tensor.size()
    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size,
                             stride=patch_size)
    # Apply Unfold to the image tensor
    unfolded = unfold(image_tensor)
    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded


# sinusoidal positional embeds
import math
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Define a module for attention blocks
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, masking=True):
        super(AttentionBlock, self).__init__()
        self.masking = masking

        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(hidden_size,
                                                    num_heads=num_heads,
                                                    batch_first=True,
                                                    dropout=0.0)

    def forward(self, x_in, kv_in, key_mask=None):
        # Apply causal masking if enabled
        if self.masking:
            bs, l, h = x_in.shape
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        else:
            mask = None

        # Perform multi-head attention operation
        return self.multihead_attn(x_in, kv_in, kv_in, attn_mask=mask,
                                   key_padding_mask=key_mask)[0]


# Define a module for a transformer block with self-attention
# and optional causal masking
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True):
        super(TransformerBlock, self).__init__()
        self.decoder = decoder

        # Layer normalization for the input
        self.norm1 = nn.LayerNorm(hidden_size)
        # Self-attention mechanism
        self.attn1 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads,
                                    masking=masking)

        # Layer normalization for the output of the first attention layer
        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            # Self-attention mechanism for the decoder with no masking
            self.attn2 = AttentionBlock(hidden_size=hidden_size,
                                        num_heads=num_heads, masking=False)

        # Layer normalization for the output before the MLP
        self.norm_mlp = nn.LayerNorm(hidden_size)
        # Multi-layer perceptron (MLP)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4),
                                 nn.ELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))

    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
        # Perform self-attention operation
        x = self.attn1(x, x, key_mask=input_key_mask) + x
        x = self.norm1(x)

        # If decoder, perform additional cross-attention layer
        if self.decoder:
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)

        # Apply MLP and layer normalization
        x = self.mlp(x) + x
        return self.norm_mlp(x)


# Define a decoder module for the Transformer architecture
class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()

        # Create an embedding layer for tokens
        self.embedding = nn.Embedding(num_emb, hidden_size)
        # Initialize the embedding weights
        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        # Initialize sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads,
                             decoder=True) for _ in range(num_layers)
        ])

        # Define a linear layer for output prediction
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None,
                encoder_padding_mask=None):
        # Embed the input sequence
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb

        # Pass the embeddings through each transformer block
        for block in self.blocks:
            embs = block(embs,
                         input_key_mask=input_padding_mask,
                         cross_key_mask=encoder_padding_mask,
                         kv_cross=encoder_output)

        return self.fc_out(embs)


# Define an Vision Encoder module for the Transformer architecture
class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128,
                 num_layers=3, num_heads=4):
        super(VisionEncoder, self).__init__()

        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)

        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length,
                                                      hidden_size).normal_(std=0.02))

        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads,
                             decoder=False, masking=False) for _ in range(num_layers)
        ])

    def forward(self, image):
        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)

        # Add a unique embedding to each token embedding
        embs = patch_emb + self.pos_embedding

        # Pass the embeddings through each transformer block
        for block in self.blocks:
            embs = block(embs)

        return embs


# the Encoder-Decoder Transformer
class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in,
                 num_emb, patch_size=16,
                 hidden_size=128, num_layers=(3, 3),
                 num_heads=4):
        super(VisionEncoderDecoder, self).__init__()
        # Create an encoder and decoder with specified parameters
        self.encoder = VisionEncoder(
            image_size=image_size, channels_in=channels_in,
            patch_size=patch_size, hidden_size=hidden_size,
            num_layers=num_layers[0], num_heads=num_heads)
        self.decoder = Decoder(num_emb=num_emb,
                               hidden_size=hidden_size,
                               num_layers=num_layers[1], num_heads=num_heads)

    def forward(self, input_image, target_seq, padding_mask):
        # Generate padding masks for the target sequence
        bool_padding_mask = padding_mask == 0
        # Encode the input sequence
        encoded_seq = self.encoder(image=input_image)
        # Decode the target sequence using the encoded sequence
        decoded_seq = self.decoder(input_seq=target_seq,
                                   encoder_output=encoded_seq,
                                   input_padding_mask=bool_padding_mask)
        return decoded_seq


def prepare_dataset():
    with open('files/dataset_flickr8k.json', 'r') as fb:
        data = json.load(fb)
        train_image_paths = []
        train_image_captions = []
        test_image_paths = []
        test_image_captions = []
        word_freq = Counter()

        max_len = 50
        for img in data['images']:
            captions = []
            for c in img['sentences']:
                word_freq.update(c['tokens'])
                if len(c['tokens']) <= max_len:
                    captions.append(c['tokens'])
            if len(captions) == 0:
                continue
            path = "files/Images/" + img['filename']
            if img['split'] in {'train', 'val', 'restval'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)
            elif img['split'] in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)

        assert len(train_image_paths) == len(train_image_captions)
        assert len(test_image_paths) == len(test_image_captions)
        print(f"there are {len(train_image_paths)} training images")
        print(f"there are {len(test_image_paths)} test images")
        # Create a dictionary to map tokens to indexes
        min_word_freq = 0
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]

        # EN tokens to indexes
        word2idx = {k: v + 4 for v, k in enumerate(words)}
        # hard code in the special tokens
        word2idx['<pad>'] = 0
        word2idx['<start>'] = 1
        word2idx['<end>'] = 2
        word2idx['<unk>'] = 3

        # Indexes to EN tokens:  translates a sequence of numbers back to tokens
        idx2word = {v: k for k, v in word2idx.items()}
        print(f"there are {len(idx2word)} unique tokens")

        train_set = FlickrD(train_image_paths,
                            train_image_captions, word2idx)
        test_set = FlickrD(test_image_paths,
                           test_image_captions, word2idx)

        train_loader = DataLoader(train_set,
                                  batch_size=128,
                                  shuffle=True)
        test_loader = DataLoader(test_set,
                                 batch_size=128,
                                 shuffle=True)

        test_images, test_tokens, \
            test_targets, test_mask = next(iter(test_loader))

        torch.save((test_images, test_tokens), "files/tests.pt")

    return word2idx, idx2word, train_loader, test_images, test_tokens


def caption(image, caption_model, device, idx2word, temp=1.0):
    # Add the Start-Of-Sentence token to the prompt
    sos_token = 1 * torch.ones(1, 1).long()
    log_tokens = [sos_token]
    caption_model.eval()
    with torch.no_grad():
        # Encode the input image
        image_embedding = caption_model.encoder(image.to(device))
        # Generate the caption tokens
        for i in range(50):
            input_tokens = torch.cat(log_tokens, 1)
            # Decode input tokens into the next predicted tokens
            data_pred = caption_model.decoder(
                input_tokens.to(device), image_embedding)
            # Sample from the distribution based on temperature
            dist = Categorical(logits=data_pred[:, -1] / temp)
            next_tokens = dist.sample().reshape(1, 1)
            # Append the next predicted token to the sequence
            log_tokens.append(next_tokens.cpu())
            # Stop if the End-Of-Caption token is predicted
            if next_tokens.item() == 2:
                break
    # Convert the list of token indices to a tensor
    pred_text = torch.cat(log_tokens, 1)
    pred_text_strings = [idx2word.get(i, "<unk>") for
                         i in pred_text[0].tolist() if i > 3]
    # Join the token strings to form the predicted text
    pred_text = " ".join(pred_text_strings)
    return pred_text


def compare(images, captions, index, caption_model, device, idx2word, temp=1.0):
    image = images[index].unsqueeze(0)
    capi = captions[index]
    capt = [idx2word.get(i, "<unk>") for i in capi.tolist() if i > 3]
    cap = " ".join(capt)
    pred = caption(image, caption_model, device, idx2word, temp=temp)
    out = torchvision.utils.make_grid(image, 1, normalize=True)
    plt.figure(figsize=(5, 10), dpi=100)
    out = torchvision.utils.make_grid(image, 1, normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.title(
        f"Original caption:\n" + cap + "\nGenerated caption:\n" + pred,
        wrap=True, loc="left", fontsize=18)
    plt.axis("off")
    plt.savefig(f'files/caption-{index}.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_model(device, word2idx):
    hidden_size = 192
    # Number of Transformer blocks for the (Encoder, Decoder)
    num_layers = (6, 6)
    # Multi-head Attention
    num_heads = 8
    # Size of the patches
    patch_size = 8
    caption_model = VisionEncoderDecoder(
        image_size=128, channels_in=3,
        num_emb=len(word2idx), patch_size=patch_size,
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads).to(device)
    return caption_model


def train(device, caption_model, train_loader):
    optimizer = torch.optim.Adam(caption_model.parameters(),
                                 lr=0.0001)
    scaler = torch.amp.GradScaler("cuda")
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Track training loss per epoch
    train_losses = []


    num_model_params = 0
    for param in caption_model.parameters():
        num_model_params += param.flatten().shape[0]
    print(f"This model has {num_model_params} parameters")

    # Iterate over epochs
    for epoch in range(0, 50):
        # Set the model in training mode
        caption_model.train()
        eloss = 0
        # Iterate over the training data loader
        for images, inputs, outputs, masks in tqdm(train_loader):
            images = images.to(device)
            tokens_in = inputs.to(device)
            padding_mask = masks.to(device)
            target_ids = outputs.to(device)
            with torch.amp.autocast("cuda"):
                # Forward pass
                pred = caption_model(images, tokens_in,
                                     padding_mask=padding_mask)
                # Compute the loss
                loss = (loss_fn(pred.transpose(1, 2),
                                target_ids) * padding_mask).mean()
                # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Log the training loss
            eloss += loss.item()
        avg_loss = eloss / len(train_loader)
        print(f"epoch {epoch}, loss is {avg_loss:.4f}")
        train_losses.append(avg_loss)

    # Save a loss curve figure and CSV
    try:
        import matplotlib.pyplot as plt
        import os

        os.makedirs("files", exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.savefig('files/loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        # Also save CSV
        with open('files/train_losses.csv', 'w') as f:
            f.write('epoch,loss')
            for i, v in enumerate(train_losses, 1):
                f.write(f"{i},{v}")
            print('Saved loss plot to files/loss_curve.png and CSV to files/train_losses.csv')
    except Exception as e:
        print('Warning: failed to save loss curve:', e)

    torch.save(caption_model.state_dict(), "files/caption.pth")
    return caption_model


def inference(caption_model, device, test_images, test_tokens, idx2word):
    caption_model.load_state_dict(torch.load("files/caption.pth",
                                             weights_only=True,
                                             map_location=device))

    for index in range(10):
        compare(test_images, test_tokens, index, caption_model, device, idx2word, temp=0.75)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    word2idx, idx2word, train_loader, test_images, test_tokens = prepare_dataset()
    caption_model = generate_model(device, word2idx)
    train(device, caption_model, train_loader)
    inference(caption_model, device, test_images, test_tokens, idx2word)

