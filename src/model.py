from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL
import torch

# Deepseek used for code assistance
# Make sure you're on the GPU runtime 

def load_models():
    """Load both the diffusion pipeline and VAE model"""

    # Determine device, cuda will always be ran on Colab
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        scheduler=DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    )
    pipe = pipe.to(device)

    # Load pretrained VAE
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae = vae.to(device)
    
    return pipe, vae, device