"""
Base Model Loading and Management

This module handles loading the base Stable Diffusion model and VAE components
with optimal settings for different hardware configurations.
"""

from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL
import torch

def load_models():
    """
    Load the base Stable Diffusion model and VAE with optimal settings.
    
    Automatically detects available hardware and configures models accordingly:
    - Uses half-precision (float16) on GPU for memory efficiency
    - Uses full precision (float32) on CPU for compatibility
    - Enables attention slicing to reduce memory usage
    
    Returns:
        tuple: (pipe, device, vae) - Pipeline, device, and VAE components
    """
    # Detect available compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use half-precision on GPU for memory efficiency, full precision on CPU for compatibility
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the main Stable Diffusion pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Official Stable Diffusion 1.5 checkpoint
        torch_dtype=torch_dtype,           # Optimize data type for current device
        scheduler=DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
    ).to(device)
    print("models 1/2 loaded")
    
    # Enable attention slicing to reduce memory usage during inference
    pipe.enable_attention_slicing()

    # Load VAE for latent space operations
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",  # VAE from SD 1.4 (compatible with 1.5)
        subfolder="vae",
        torch_dtype=torch_dtype
    ).to(device)
    print("models 2/2 loaded")

    return pipe, device, vae