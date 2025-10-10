from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

import torch.nn.functional as F


def preprocess_image(image):
    if image.mode != 'RGB':
            image = image.convert('RGB')
    transform = T.Compose([
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor()
    ])
    x = transform(image).unsqueeze(0)
    return x


def add_noise_to_image(image, strength):
    #calculate noise injection by chosen strength
    noise_val = strength * 95 #assumes noise is between 0.1 - 1.0
    # Single conversion to numpy array
    img_array = np.array(image)
    noise = np.random.normal(0, noise_val, img_array.shape).astype(np.int16)
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


def get_latent_tensor(vae, image, device):
    if isinstance(image, Image.Image):
        image = preprocess_image(image)  # Convert PIL to tensor if needed
    with torch.no_grad():
        latent_tensor = vae.encode(image.to(device)).latent_dist.sample() * 0.18215 #scale factor used in SD
    
    return latent_tensor  # Returns [1, 4, 48, 48] tensor


def visualize_latent_tensor(latent_tensor):
    """Convert a latent tensor to a channel overlay visualization"""
    latent = latent_tensor[0]  # Remove batch dimension [4, 48, 48]

    # Get the actual dimensions of the latent tensor
    num_channels, height, width = latent.shape
    
    # Normalize each channel to [0, 1] range independently
    normalized_channels = []
    for i in range(4):
        channel = latent[i]
        if channel.max() > channel.min():
            channel_normalized = (channel - channel.min()) / (channel.max() - channel.min())
        else:
            channel_normalized = torch.zeros_like(channel)
        normalized_channels.append(channel_normalized)
    
    # Assign each channel a distinct color for clear differentiation
    colors = [
        [1.0, 0.0, 0.0],  # Red - Channel 0
        [0.0, 1.0, 0.0],  # Green - Channel 1  
        [0.0, 0.0, 1.0],  # Blue - Channel 2
        [1.0, 1.0, 0.0]   # Yellow - Channel 3
    ]
    
    # Create RGB composite by summing colored channels
    composite = torch.zeros(3, height, width)
    for i in range(4):
        color_weight = torch.tensor(colors[i]).view(3, 1, 1).to(latent_tensor.device)
        composite += normalized_channels[i] * color_weight
    
    # Normalize final composite to prevent over-saturation
    composite = composite.clamp(0, 1)
    
    # Convert to PIL Image
    overlay_image = T.ToPILImage()(composite.cpu())
    
    output_size=(192, 192)
    overlay_image = overlay_image.resize(output_size, Image.Resampling.LANCZOS)
    
    return overlay_image


def latent_channel_vis(vae, image, device):
    latent_tensor = get_latent_tensor(vae, image, device)
    return visualize_latent_tensor(latent_tensor)


def generate_with_progression(pipe, image, prompt, device, strength, guidance_scale, num_inference_steps):
    # Calculate the actual number of steps that will be run by the pipeline.
    actual_num_steps = int(num_inference_steps * strength)
    if actual_num_steps < 1:
        # If strength is so low that no steps are run, return an error or default.
        # For this app, returning the original image and empty steps is safest.
        print("Strength is too low, resulting in zero steps. Skipping generation.")
        return image, []

    denoising_steps = []
    # Dynamically determine how many images to capture.
    num_to_capture = min(4, actual_num_steps)
    # Base the capture points on the actual number of steps.
    capture_steps = np.linspace(0, actual_num_steps - 1, num_to_capture, dtype=int).tolist()
    print(f"Actual steps to run: {actual_num_steps}. Capturing latents at steps: {capture_steps}")

    def callback(pipe, step_index, timestep, callback_kwargs):
        if step_index in capture_steps:
            latents = callback_kwargs["latents"]
            denoising_steps.append(latents.clone())
        return callback_kwargs

    if not hasattr(pipe, '_vae_slicing_enabled') or not pipe._vae_slicing_enabled:
        pipe.vae.enable_slicing()
        pipe._vae_slicing_enabled = True
    
    pipeline_output = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device=device).manual_seed(42),
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        safety_checker=None
    )
    result = pipeline_output.images[0]

    denoising_visualizations = [visualize_latent_tensor(latent) for latent in denoising_steps]

    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result, denoising_visualizations



def create_denoising_collage(denoising_steps):
    if not denoising_steps:
        # Return a blank image if the list is empty for any reason
        return Image.new('RGB', (384, 384), color = 'grey')

    # Resize all to same size for consistent grid
    resized_steps = [step.resize((192, 192), Image.Resampling.LANCZOS) for step in denoising_steps]
    
    # Create 2x2 grid
    collage = Image.new('RGB', (384, 384), color = 'black')
    
    positions = [(0, 0), (192, 0), (0, 192), (192, 192)]
    
    # Safely paste each available image into its slot
    for i, img in enumerate(resized_steps):
        if i < len(positions):
            collage.paste(img, positions[i])
            
    return collage




#maybe??
def visualize_prompt_guidance(pipe, prompt, device):
    """
    Generates a visual representation of the text embedding for a prompt.
    """
    # 1. Get the text embeddings from the pipeline's text encoder
    text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    # 2. Normalize and reshape the vector for visualization
    # The output is [1, 77, 768]. We'll visualize the 77x768 tensor.
    embedding_tensor = text_embeddings.squeeze(0) # Shape: [77, 768]
    
    # Normalize the tensor to a 0-1 range to be used as colors
    embedding_tensor = (embedding_tensor - embedding_tensor.min()) / (embedding_tensor.max() - embedding_tensor.min())

    # 3. Convert the tensor to a PIL image
    # We need to add a channel dimension for grayscale to RGB conversion
    heatmap_pil = T.ToPILImage()(embedding_tensor.unsqueeze(0).cpu())
    
    # 4. Colorize and resize for better visual appeal
    heatmap_pil = heatmap_pil.resize((384, 150), Image.Resampling.NEAREST) # Resize with sharp pixels
    heatmap_colored = heatmap_pil.convert("RGB") # Can apply colormaps here if desired

    return heatmap_colored