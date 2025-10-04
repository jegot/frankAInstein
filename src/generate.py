from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np


def preprocess_image(image):
    """Load and preprocess image"""
    if image.mode != 'RGB':
            image = image.convert('RGB')
    transform = T.Compose([
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor()
    ])
    x = transform(image).unsqueeze(0)
    return x


def encode_image(vae, image, device):
    if isinstance(image, Image.Image):
        image = preprocess_image(image)  # Convert PIL to tensor if needed

    with torch.no_grad():
        latent = vae.encode(image.to(device)).latent_dist.sample() * 0.18215  # scale factor used in SD
        reconstructed = vae.decode(latent).sample

    # Convert back to image
    reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
    reconstructed_image = T.ToPILImage()(reconstructed.squeeze().cpu())

    # Clean up GPU memory
    del latent, reconstructed
    if device == "cuda":
        torch.cuda.empty_cache()

    # Resize it to mimic latent space enconding
    new_size = (128, 128)  # diffusion model typically expects 512x512, but we'll go smaller for visualization purposes 
    reconstructed_image = reconstructed_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return reconstructed_image


def add_noise_to_image(image):
    # Single conversion to numpy array
    img_array = np.array(image)  # Default uint8, no dtype conversion needed
    # Generate noise directly as uint8 with proper scaling
    noise = np.random.normal(0, 25, img_array.shape).astype(np.int16)
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array)



def generate_style_transfer(pipe, image, prompt, device, strength, guidance_scale, num_inference_steps):
    """Generate style transfer using the diffusion pipeline"""
    # Only enable VAE slicing for memory savings (works on both CPU/GPU)
    if not hasattr(pipe, '_vae_slicing_enabled') or not pipe._vae_slicing_enabled:
        pipe.vae.enable_slicing()
        pipe._vae_slicing_enabled = True
    
    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(42),
        output_type="pil" 
    ).images[0]

    # Cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result




def check_nsfw_content(image):
    #Check if image is all black (NSFW filter triggered)
    img_array = np.array(image)
    black_pixels = np.sum(img_array < 10)
    total_pixels = img_array.size
    black_ratio = black_pixels / total_pixels
    return black_ratio > 0.95


def create_side_by_side(before_img, after_img):
            before_resized = before_img.resize((384, 384), Image.Resampling.LANCZOS)
            after_resized = after_img.resize((384, 384), Image.Resampling.LANCZOS)
            comparison_img = Image.new('RGB', (768, 384))
            comparison_img.paste(before_resized, (0, 0))
            comparison_img.paste(after_resized, (384, 0))
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


def tensor_to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # Clamp and normalize to [0, 1] if needed
    tensor = tensor.detach().cpu().clamp(0, 1)
    return T.ToPILImage()(tensor)
