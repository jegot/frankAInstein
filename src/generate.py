from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from .model import load_models



def preprocess_image(image):
    """Load and preprocess image"""
    transform = T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # VAE expects [-1, 1]
    ])
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 512, 512]
    return x

def encode_image(vae, x, device):
    """Encode image to latent space"""
    with torch.no_grad():
        latent = vae.encode(x.to(device)).latent_dist.sample() * 0.18215  # scale factor used in SD
        reconstructed = vae.decode(latent).sample

    # Convert back to image
    reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
    reconstructed_image = T.ToPILImage()(reconstructed.squeeze().cpu())

    # Resize it to mimic latent space enconding
    new_size = (256, 256)  # diffusion model typically expects 512x512, but we'll go smaller for visualization purposes 
    reconstructed_image = reconstructed_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return reconstructed_image, latent


def add_noise_to_image(image):
    """Convert to numpy and add noise"""
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, 0.3, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 1)
    noisy_img = Image.fromarray((noisy_array * 255).astype(np.uint8))
    return noisy_img

def generate_style_transfer(pipe, image, prompt, device, strength=0.5, guidance_scale=5, num_inference_steps=25):
    """Generate style transfer using the diffusion pipeline"""
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(42),  # For reproducibility
        eta=0.0,  # DDIM eta parameter
        output_type="pil"  # Ensure PIL output
    ).images[0]
    
    
    result.save("final_generation.png")
    
    return result