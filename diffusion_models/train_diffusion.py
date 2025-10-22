import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
IMG_SIZE = 64
CHANNELS = 3
TIMESTEPS = 1000
LEARNING_RATE = 2e-4
EPOCHS = 100
SAVE_INTERVAL = 10

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.group_norm(x)).chunk(3, dim=1)
        
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.permute(0, 2, 3, 1).view(b, h * w, c)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        
        attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c), dim=-1)
        out = torch.bmm(attention, v).view(b, h, w, c).permute(0, 3, 1, 2)
        
        return self.to_out(out) + x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.time_dim = features[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(features[0]),
            nn.Linear(features[0], self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, features[0], 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        for i in range(len(features)):
            in_ch = features[i-1] if i > 0 else features[0]
            out_ch = features[i]
            
            self.encoder_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch, out_ch, self.time_dim),
                ResidualBlock(out_ch, out_ch, self.time_dim)
            ]))
            
            self.encoder_attentions.append(
                AttentionBlock(out_ch) if i >= 2 else nn.Identity()
            )
            
            if i < len(features) - 1:
                self.downsample_blocks.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.downsample_blocks.append(nn.Identity())
        
        # Middle
        mid_dim = features[-1]
        self.middle_block1 = ResidualBlock(mid_dim, mid_dim, self.time_dim)
        self.middle_attention = AttentionBlock(mid_dim)
        self.middle_block2 = ResidualBlock(mid_dim, mid_dim, self.time_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i in reversed(range(len(features))):
            out_ch = features[i-1] if i > 0 else features[0]
            in_ch = features[i]
            
            self.decoder_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch + in_ch, in_ch, self.time_dim),
                ResidualBlock(in_ch, out_ch, self.time_dim)
            ]))
            
            self.decoder_attentions.append(
                AttentionBlock(out_ch) if i-1 >= 2 else nn.Identity()
            )
            
            if i > 0:
                self.upsample_blocks.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            else:
                self.upsample_blocks.append(nn.Identity())
        
        # Final projection
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, features[0]),
            nn.SiLU(),
            nn.Conv2d(features[0], out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps):
        time_emb = self.time_mlp(timesteps)
        
        x = self.init_conv(x)
        
        # Encoder
        encoder_outputs = []
        for i, (blocks, attention, downsample) in enumerate(zip(
            self.encoder_blocks, self.encoder_attentions, self.downsample_blocks
        )):
            x = blocks[0](x, time_emb)
            x = blocks[1](x, time_emb)
            x = attention(x)
            encoder_outputs.append(x)
            x = downsample(x)
        
        # Middle
        x = self.middle_block1(x, time_emb)
        x = self.middle_attention(x)
        x = self.middle_block2(x, time_emb)
        
        # Decoder
        for i, (blocks, attention, upsample) in enumerate(zip(
            self.decoder_blocks, self.decoder_attentions, self.upsample_blocks
        )):
            skip = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = blocks[0](x, time_emb)
            x = blocks[1](x, time_emb)
            x = attention(x)
            x = upsample(x)
        
        return self.final_conv(x)

class DiffusionTrainer:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        
        # Define beta schedule
        self.betas = self.linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t):
        """Calculate loss for training"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

def prepare_data():
    """Prepare CelebA dataset"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Download CelebA dataset
    dataset = CelebA(
        root='./data',
        split='train',
        target_type='attr',
        transform=transform,
        download=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

@torch.no_grad()
def sample_images(diffusion_trainer, model, num_samples=8):
    """Generate samples from the trained model"""
    model.eval()
    
    # Start from pure noise
    img = torch.randn(num_samples, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    
    for i in reversed(range(diffusion_trainer.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(img, t)
        
        # Remove noise
        alpha = diffusion_trainer.alphas[i]
        alpha_cumprod = diffusion_trainer.alphas_cumprod[i]
        beta = diffusion_trainer.betas[i]
        
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)
        
        # Denoise step
        img = (1 / torch.sqrt(alpha)) * (
            img - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise
    
    # Unnormalize images
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    
    return img

def train_model():
    """Main training function"""
    # Prepare data
    print("Preparing dataset...")
    dataloader = prepare_data()
    print(f"Dataset loaded with {len(dataloader.dataset)} samples")
    
    # Initialize model
    model = UNet().to(device)
    diffusion_trainer = DiffusionTrainer(model, TIMESTEPS)
    
    # Move diffusion parameters to device
    diffusion_trainer.betas = diffusion_trainer.betas.to(device)
    diffusion_trainer.alphas = diffusion_trainer.alphas.to(device)
    diffusion_trainer.alphas_cumprod = diffusion_trainer.alphas_cumprod.to(device)
    diffusion_trainer.sqrt_alphas_cumprod = diffusion_trainer.sqrt_alphas_cumprod.to(device)
    diffusion_trainer.sqrt_one_minus_alphas_cumprod = diffusion_trainer.sqrt_one_minus_alphas_cumprod.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Create directories for saving
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('ldm_celeba/samples', exist_ok=True)
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
            
            # Calculate loss
            loss = diffusion_trainer.p_losses(images, t)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}')
        
        # Save checkpoint and generate samples
        if (epoch + 1) % SAVE_INTERVAL == 0:
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoints/diffusion_epoch_{epoch+1}.pt')
            
            # Generate and save sample images
            sample_imgs = sample_images(diffusion_trainer, model)
            
            # Create a grid of images
            grid = torchvision.utils.make_grid(sample_imgs, nrow=4, padding=2)
            
            # Save the grid
            torchvision.utils.save_image(grid, f'diffusion_models/ldm_celeba/samples/epoch_{epoch + 1}.png')
            print(f'Saved samples for epoch {epoch+1}')
    
    print("Training completed!")
    return model, diffusion_trainer

def load_and_sample(checkpoint_path, num_samples=16):
    """Load a trained model and generate samples"""
    model = UNet().to(device)
    diffusion_trainer = DiffusionTrainer(model, TIMESTEPS)
    
    # Move diffusion parameters to device
    diffusion_trainer.betas = diffusion_trainer.betas.to(device)
    diffusion_trainer.alphas = diffusion_trainer.alphas.to(device)
    diffusion_trainer.alphas_cumprod = diffusion_trainer.alphas_cumprod.to(device)
    diffusion_trainer.sqrt_alphas_cumprod = diffusion_trainer.sqrt_alphas_cumprod.to(device)
    diffusion_trainer.sqrt_one_minus_alphas_cumprod = diffusion_trainer.sqrt_one_minus_alphas_cumprod.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Generate samples
    sample_imgs = sample_images(diffusion_trainer, model, num_samples)
    
    # Display samples
    grid = torchvision.utils.make_grid(sample_imgs, nrow=4, padding=2)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title('Generated Samples')
    plt.show()
    
    return model, diffusion_trainer

if __name__ == "__main__":
    # Train the model
    model, diffusion_trainer = train_model()
    
    # Generate final samples
    final_samples = sample_images(diffusion_trainer, model, 16)
    grid = torchvision.utils.make_grid(final_samples, nrow=4, padding=2)
    torchvision.utils.save_image(grid, 'final_samples.png')
    print("Final samples saved as 'final_samples.png'")