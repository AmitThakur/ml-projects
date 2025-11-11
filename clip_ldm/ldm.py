import sys

sys.path.append("pytorch-stable-diffusion/sd")

from model_converter import load_from_standard_weights
from encoder import VAE_Encoder
from decoder import VAE_Decoder
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPTokenizer
from clip import CLIP
from diffusion import Diffusion
from ddpm import DDPMSampler
from tqdm import tqdm
from copy import deepcopy
import pipeline
import model_loader


def center_crop(img):
    width, height = img.size
    new = min(width, height)
    left = (width - new) / 2
    top = (height - new) / 2
    right = (width + new) / 2
    bottom = (height + new) / 2
    im = img.crop((left, top, right, bottom))
    return im


def trial(encoder, decoder):
    image1 = center_crop(Image.open("files/amit_hd.jpeg")).resize((512, 512))
    image2 = center_crop(Image.open("files/ucm.jpeg")).resize((512, 512))
    transform = transforms.ToTensor()
    image1 = transform(image1)
    image2 = transform(image2)
    print(image1.shape, image2.shape)
    N = torch.distributions.Normal(0, 1)
    noise = N.sample(torch.Size([1, 4, 64, 64]))
    latent1 = encoder(image1.unsqueeze(0), noise)[0]
    latent2 = encoder(image2.unsqueeze(0), noise)[0]
    print(latent1.shape, latent2.shape)
    imgs = [image1.permute(1, 2, 0),
            torch.clip(latent1.permute(1, 2, 0), -1, 1).detach().cpu().numpy()
            .repeat(8, axis=0).repeat(8, axis=1) / 2 + 0.5,
            image2.permute(1, 2, 0), torch.clip(latent2.permute(1, 2, 0), -1, 1).detach().cpu()
            .numpy().repeat(8, axis=0).repeat(8, axis=1) / 2 + 0.5]
    plt.figure(figsize=(12, 3), dpi=100)
    for i in range(4):  # B
        plt.subplot(1, 4, i + 1)
        plt.imshow(imgs[i])
        if i % 2 == 0:
            plt.title("Original Image")
        else:
            plt.title("Latent Image")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("files/vae_encoded.png")
    plt.close()

    reconstruct1 = decoder(latent1.unsqueeze(0))
    reconstruct2 = decoder(latent2.unsqueeze(0))
    print(reconstruct1.shape, reconstruct2.shape)
    imgs = [image1.permute(1, 2, 0),
            torch.clip(reconstruct1[0].permute(1, 2, 0),
                       0, 1).detach().cpu().numpy(),
            image2.permute(1, 2, 0),
            torch.clip(reconstruct2[0].permute(1, 2, 0),
                       0, 1).detach().cpu().numpy()]
    plt.figure(figsize=(12, 3), dpi=100)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imgs[i])
        if i % 2 == 0:
            plt.title("Original Image")
        else:
            plt.title("Reconstructed Image")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("files/vae_decoded.png")
    plt.close()


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0,
                                           end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep],
                     dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def evaluate():
    model_file = "files/hollie-mengert.ckpt"
    state_dict = load_from_standard_weights(model_file, "cpu")
    encoder = VAE_Encoder()
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    decoder = VAE_Decoder()
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    # trial(encoder, decoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder.to(device)
    tokenizer = CLIPTokenizer("files/vocab.json", merges_file="files/merges.txt")
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    prompt = '''A dog typing on a computer, wearing glasses and a jacket, highly detailed, ultra sharp, cinematic, 
    100mm lens, 8k resolution.'''

    uncond_prompt = ""

    seed = 42
    with torch.no_grad():
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77).input_ids
        cond_tokens = torch.tensor(cond_tokens,
                                   dtype=torch.long, device=device)
        print("conditional tokens are\n", cond_tokens)
        cond_context = clip(cond_tokens).to(device)
        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],
                                                    padding="max_length", max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens,
                                     dtype=torch.long, device=device)
        print("unconditional tokens are\n", uncond_tokens)
        uncond_context = clip(uncond_tokens).to(device)
        clip.to("cpu")
        context = torch.cat([cond_context, uncond_context])

        diffusion = Diffusion().to(device)
        diffusion.load_state_dict(state_dict['diffusion'], strict=True)

        num_reference_steps = 50
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(num_reference_steps)
        cfg_scale = 8
        latent_shape = (1, 4, 64, 64)
        noisy_latents = []
        with torch.no_grad():
            latents = torch.randn(latent_shape, generator=generator, device=device)
            timesteps = tqdm(sampler.timesteps)
            for i, timestep in enumerate(timesteps):
                time_embedding = get_time_embedding(timestep).to(device)
                model_input = latents.to(device)
                model_input = model_input.repeat(2, 1, 1, 1)
                model_output = diffusion(model_input, context, time_embedding)
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + \
                               output_uncond
                latents = sampler.step(timestep, latents, model_output)
                if i % 10 == 9:
                    noisy_latents.append(deepcopy(latents))
        diffusion.to("cpu")

        final_latent = noisy_latents[-1]
        final_output = decoder(final_latent)
        img = torch.clip(final_output[0].permute(1, 2, 0) / 2 + 0.5, 0, 1)
        plt.figure(dpi=100)
        plt.imshow(img.detach().cpu().numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("files/sample_dog.png")

        plt.figure(figsize=(10, 3), dpi=100)
        for i in range(5):
            im = torch.clip(noisy_latents[i][0].permute(1, 2, 0) / 2 + 0.5, 0, 1)
            plt.subplot(1, 5, i + 1)
            plt.imshow(im.detach().cpu().numpy())
            plt.title(f"Latent Image\nat t={800 - i * 200}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("files/Latent.png")

        plt.figure(figsize=(10, 3), dpi=100)
        for i in range(5):
            im = decoder(noisy_latents[i])
            im = torch.clip(im[0].permute(1, 2, 0) / 2 + 0.5, 0, 1)
            plt.subplot(1, 5, i + 1)
            plt.imshow(im.detach().cpu().numpy())
            plt.title(f"Decoded Image\nat t={800 - i * 200}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("files/Decoded_series.png")


def modify_image(prompt):
    model_file = "files/hollie-mengert.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = model_loader.preload_models_from_standard_weights(model_file, device)
    tokenizer = CLIPTokenizer("files/vocab.json", merges_file="files/merges.txt")
    uncond_prompt = ""
    input_image = Image.open("files/ucm.jpeg")

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=0.8,  # 0 to 1.0: a higher value means the output deviates more from the original image
        do_cfg=True,
        cfg_scale=1,  # Higher toward text condition
        sampler_name="ddpm",
        n_inference_steps=50,
        seed=42,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.imshow(output_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('files/modified_image.png')


if __name__ == '__main__':
    # evaluate()
    modify_image("Ducks walking on the grass of the campus")