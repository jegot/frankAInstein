from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
#from .model import load_models

def preprocess_image(image):
    """Load and preprocess image"""
    if image.mode != 'RGB':
            image = image.convert('RGB')
    transform = T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # VAE expects [-1, 1]
    ])
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 512, 512]
    return x


def encode_image(vae, image, device):
    with torch.no_grad():
        latent = vae.encode(image.to(device)).latent_dist.sample() * 0.18215  # scale factor used in SD
        reconstructed = vae.decode(latent).sample

    # Convert back to image
    reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
    reconstructed_image = T.ToPILImage()(reconstructed.squeeze().cpu())

    # Resize it to mimic latent space enconding
    new_size = (256, 256)  # diffusion model typically expects 512x512, but we'll go smaller for visualization purposes 
    reconstructed_image = reconstructed_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return reconstructed_image


def add_noise_to_image(image):
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, 0.3, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 1)
    noisy_img = Image.fromarray((noisy_array * 255).astype(np.uint8))
    return noisy_img


def generate_style_transfer(pipe, image, prompt, device, strength, guidance_scale, num_inference_steps):
    """Generate style transfer using the diffusion pipeline"""
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(42),
        output_type="pil" 
    ).images[0]
    return result


def check_nsfw_content(image):
    #Check if image is all black (NSFW filter triggered)
    img_array = np.array(image)
    black_pixels = np.sum(img_array < 10)
    total_pixels = img_array.size
    black_ratio = black_pixels / total_pixels
    return black_ratio > 0.95


def create_side_by_side(before_img, after_img):
            before_resized = before_img.resize((512, 512), Image.Resampling.LANCZOS)
            after_resized = after_img.resize((512, 512), Image.Resampling.LANCZOS)
            comparison_img = Image.new('RGB', (1024, 512))
            comparison_img.paste(before_resized, (0, 0))
            comparison_img.paste(after_resized, (512, 0))
            return comparison_img



def prompt_conversion(style):
    style_prompts = {
        "anime style": "rendered in anime style",
        "cartoon style": "illustrated in cartoon style",
        "van gogh style": "in the style of Van Gogh",
        "watercolor painting": "as a watercolor painting",
        "pixel art": "in pixel art format",
        "sketch drawing": "as a pencil sketch",
        "oil painting": "in oil painting style",
        "cyberpunk style": "with cyberpunk visual aesthetics",
        "vintage poster": "as a vintage poster design"
    }
    # Default fallback if style not found
    full_prompt = style_prompts.get(style, "stylized rendering")
    return full_prompt
