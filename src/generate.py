"""
Image Processing and Generation Utilities

This module contains core functions for image preprocessing, latent space operations,
and the complete diffusion pipeline with step-by-step visualization capabilities.
"""

from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F

def preprocess_image(image):
    """
    Preprocess input image for the diffusion pipeline.
    
    Converts image to RGB format, resizes to 384x384, and normalizes to [0,1] range.
    This ensures consistent input format for the Stable Diffusion model.
    
    Args:
        image: PIL Image - Input image to preprocess
        
    Returns:
        torch.Tensor: Preprocessed image tensor [1, 3, 384, 384]
    """
    # Ensure RGB format for consistent processing
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define preprocessing pipeline
    transform = T.Compose([
        T.Resize(384),        # Resize to standard size
        T.CenterCrop(384),    # Center crop to square
        T.ToTensor()          # Convert to tensor and normalize to [0,1]
    ])
    
    # Apply transforms and add batch dimension
    x = transform(image).unsqueeze(0)
    return x

def add_noise_to_image(image, strength):
    """
    Add controlled noise to an image for visualization purposes.
    
    This function simulates the noise injection step in the diffusion process.
    The amount of noise is proportional to the strength parameter.
    
    Args:
        image: PIL Image - Input image
        strength: float - Noise strength (0.3-1.0)
        
    Returns:
        PIL Image: Image with added noise
    """
    # Calculate noise amount based on strength parameter
    noise_val = strength * 95
    
    # Convert to numpy array for noise addition
    img_array = np.array(image)
    
    # Generate random noise with normal distribution
    noise = np.random.normal(0, noise_val, img_array.shape).astype(np.int16)
    
    # Add noise and clip to valid range [0, 255]
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array)

def check_nsfw_content(image):
    """
    Simple content detection based on image characteristics.
    
    This function detects potentially problematic content by checking for
    mostly black images, which often indicate failed generations.
    
    Args:
        image: PIL Image - Image to check
        
    Returns:
        bool: True if potentially inappropriate content detected
    """
    img_array = np.array(image)
    
    # Count very dark pixels (potential black screen)
    black_pixels = np.sum(img_array < 10)
    total_pixels = img_array.size
    black_ratio = black_pixels / total_pixels
    
    # Flag if image is mostly black (common in failed generations)
    return black_ratio > 0.95

def create_side_by_side(before_img, after_img):
    """
    Create a side-by-side comparison of before and after images.
    
    Args:
        before_img: PIL Image - Original image
        after_img: PIL Image - Transformed image
        
    Returns:
        PIL Image: Side-by-side comparison image
    """
    # Resize both images to same size for comparison
    before_resized = before_img.resize((384, 384), Image.Resampling.LANCZOS)
    after_resized = after_img.resize((384, 384), Image.Resampling.LANCZOS)
    
    # Create new image with double width
    comparison_img = Image.new('RGB', (768, 384))
    
    # Paste images side by side
    comparison_img.paste(before_resized, (0, 0))
    comparison_img.paste(after_resized, (384, 0))
    
    return comparison_img

def prompt_conversion(style):
    """
    Convert style names to detailed prompts for the diffusion model.
    
    This function maps user-selected style names to comprehensive text prompts
    that guide the diffusion process. Includes prompts for both base styles
    and fine-tuned models with enhanced descriptions.
    
    Args:
        style: str - Style name selected by user
        
    Returns:
        str: Detailed prompt for the diffusion model
    """
    style_prompts = {
        # Base model styles
        "anime style": "anime artwork, vibrant colors, clean line art, highly detailed, studio anime aesthetic",
        "van gogh style": "painting in the style of Vincent van Gogh, swirling brush strokes, post-impressionist texture",
        "watercolor painting": "delicate watercolor painting, soft brush strokes, light washes, flowing pigments",
        "pixel art": "8-bit pixel art, blocky low-resolution graphics, retro video game aesthetic",
        "3D animation": "cinematic 3D animation, in the style of Pixar, appealing characters, detailed textures, sophisticated lighting",
        "cyberpunk style": "cyberpunk artwork, neon lights, futuristic cityscape, high-tech dystopian aesthetic",
        
        # Fine-tuned model styles (more detailed prompts)
        "ghibli": "studio ghibli style, anime artwork, whimsical scenery, highly detailed, expressive characters, soft lighting, beautiful art, hand-drawn animation aesthetic",
        "lego": "LEGO style, blocky plastic textures, modular shapes, toy aesthetic, bright primary colors, clean edges, building blocks",
        "2d_animation": "2D animation style, flat cel-shaded illustration, bold outlines, simplified shapes, expressive poses, limited color palette, comic-style clarity, hand-drawn animation aesthetic",
        "3d_animation": "3D animation style, cinematic 3D animation, in the style of Pixar, appealing characters, detailed textures, sophisticated lighting, computer-generated imagery"
    }
    
    full_prompt = style_prompts.get(style, "stylized rendering")
    return full_prompt

def tensor_to_pil(tensor):
    """
    Convert PyTorch tensor to PIL Image with proper normalization.
    
    Args:
        tensor: torch.Tensor - Image tensor
        
    Returns:
        PIL Image: Converted image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Clamp values to [0,1] range and move to CPU
    tensor = tensor.detach().cpu().clamp(0, 1)
    
    # Convert to PIL Image
    return T.ToPILImage()(tensor)

def get_latent_tensor(vae, image, device):
    """
    Encode image to latent space using the VAE encoder.
    
    Args:
        vae: VAE model for encoding
        image: PIL Image or tensor - Input image
        device: torch.device - Target device
        
    Returns:
        torch.Tensor: Latent representation
    """
    # Preprocess if PIL Image
    if isinstance(image, Image.Image):
        image = preprocess_image(image)
    
    with torch.no_grad():
        # Ensure image is on correct device with correct dtype
        image_for_vae = image.to(device=device, dtype=vae.dtype)
        
        # Encode to latent space and apply scaling factor
        latent_tensor = vae.encode(image_for_vae).latent_dist.sample() * 0.18215
    
    return latent_tensor

def visualize_latent_tensor(latent_tensor):
    """
    Convert latent tensor to a colorful channel visualization.
    
    Each of the 4 latent channels is assigned a different color (red, green, blue, yellow)
    to create an intuitive visualization of the latent space representation.
    
    Args:
        latent_tensor: torch.Tensor - Latent representation [1, 4, 48, 48]
        
    Returns:
        PIL Image: Colorized latent visualization
    """
    latent = latent_tensor[0]  # Remove batch dimension [4, 48, 48]
    
    # Get device from input tensor
    device = latent_tensor.device
    
    # Get dimensions
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
    
    # Assign distinct colors to each channel
    colors = [
        [1.0, 0.0, 0.0],  # Red - Channel 0
        [0.0, 1.0, 0.0],  # Green - Channel 1  
        [0.0, 0.0, 1.0],  # Blue - Channel 2
        [1.0, 1.0, 0.0]   # Yellow - Channel 3
    ]
    
    # Create RGB composite by summing colored channels
    composite = torch.zeros(3, height, width, device=device)
    for i in range(4):
        color_weight = torch.tensor(colors[i], device=device).view(3, 1, 1)
        composite += normalized_channels[i] * color_weight
    
    # Normalize final composite to prevent over-saturation
    composite = composite.clamp(0, 1)
    
    # Convert to PIL Image (moves to CPU automatically)
    overlay_image = T.ToPILImage()(composite.cpu())
    
    # Resize for better visibility
    output_size = (192, 192)
    overlay_image = overlay_image.resize(output_size, Image.Resampling.LANCZOS)
    
    return overlay_image

def latent_channel_vis(vae, image, device):
    """
    Wrapper function to encode image and visualize latent channels.
    
    Args:
        vae: VAE model
        image: PIL Image or tensor
        device: torch.device
        
    Returns:
        PIL Image: Latent channel visualization
    """
    latent_tensor = get_latent_tensor(vae, image, device)
    return visualize_latent_tensor(latent_tensor)

def generate_with_progression(pipe, image, prompt, device, strength, guidance_scale, num_inference_steps):
    """
    Generate image with step-by-step denoising progression tracking.
    
    This function runs the main diffusion process while capturing intermediate
    latent states to show the denoising progression visually.
    
    Args:
        pipe: Diffusion pipeline
        image: PIL Image - Input image
        prompt: str - Text prompt
        device: torch.device
        strength: float - Denoising strength
        guidance_scale: float - Prompt guidance strength
        num_inference_steps: int - Number of denoising steps
        
    Returns:
        tuple: (final_image, denoising_steps_visualizations)
    """
    # Calculate actual number of steps based on strength
    actual_num_steps = int(num_inference_steps * strength)
    if actual_num_steps < 1:
        print("Strength is too low, resulting in zero steps. Skipping generation.")
        return image, []

    denoising_steps = []
    num_to_capture = min(4, actual_num_steps)  # Capture up to 4 intermediate steps
    capture_steps = np.linspace(0, actual_num_steps - 1, num_to_capture, dtype=int).tolist()
    print(f"Actual steps to run: {actual_num_steps}. Capturing latents at steps: {capture_steps}")

    def callback(pipe, step_index, timestep, callback_kwargs):
        """Callback function to capture intermediate denoising steps."""
        if step_index in capture_steps:
            latents = callback_kwargs["latents"]
            denoising_steps.append(latents.clone())
        return callback_kwargs

    # Enable VAE slicing for memory efficiency
    if not hasattr(pipe, '_vae_slicing_enabled') or not pipe._vae_slicing_enabled:
        pipe.vae.enable_slicing()
        pipe._vae_slicing_enabled = True
    
    # Use CPU generator for better compatibility
    generator = torch.Generator(device="cpu").manual_seed(42)
    
    # Run the diffusion pipeline with callback
    pipeline_output = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )
    result = pipeline_output.images[0]

    # Convert captured latents to visualizations
    denoising_visualizations = [visualize_latent_tensor(latent) for latent in denoising_steps]

    # Clean up GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result, denoising_visualizations

def create_denoising_collage(denoising_steps):
    """
    Create a 2x2 collage of denoising progression steps.
    
    Args:
        denoising_steps: list of PIL Images - Intermediate denoising steps
        
    Returns:
        PIL Image: 2x2 collage of denoising progression
    """
    if not denoising_steps:
        return Image.new('RGB', (384, 384), color='grey')

    # Resize all steps to same size
    resized_steps = [step.resize((192, 192), Image.Resampling.LANCZOS) for step in denoising_steps]
    
    # Create collage canvas
    collage = Image.new('RGB', (384, 384), color='black')
    positions = [(0, 0), (192, 0), (0, 192), (192, 192)]
    
    # Paste images in 2x2 grid
    for i, img in enumerate(resized_steps):
        if i < len(positions):
            collage.paste(img, positions[i])
            
    return collage

def visualize_prompt_guidance(pipe, prompt, device):
    """
    Create a heatmap visualization of text prompt embeddings.
    
    Args:
        pipe: Diffusion pipeline
        prompt: str - Text prompt
        device: torch.device
        
    Returns:
        PIL Image: Prompt embedding heatmap
    """
    # Tokenize and encode prompt
    text_input = pipe.tokenizer(
        prompt, 
        padding="max_length", 
        max_length=pipe.tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    # Normalize embeddings for visualization
    embedding_tensor = text_embeddings.squeeze(0)
    embedding_tensor = (embedding_tensor - embedding_tensor.min()) / (embedding_tensor.max() - embedding_tensor.min())

    # Convert to PIL Image
    heatmap_pil = tensor_to_pil(embedding_tensor)
    heatmap_pil = heatmap_pil.resize((384, 150), Image.Resampling.NEAREST)
    
    return heatmap_pil