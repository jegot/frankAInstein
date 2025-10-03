from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL
import torch

def load_models():
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load Stable Diffusion pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        revision="fp16" if device == "cuda" else None,
        scheduler=DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
        )
    ).to(device)

    pipe.enable_attention_slicing()  # helps reduce memory usage

    # Load VAE for latent encoding
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        torch_dtype=torch_dtype
    ).to(device)

    return pipe, device, vae