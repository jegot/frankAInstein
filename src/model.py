import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL
from sklearn.manifold import TSNE

import torchvision.transforms as T


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    scheduler=DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
)
pipe = pipe.to("cuda")

# Load pretrained VAE
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
#vae.eval()