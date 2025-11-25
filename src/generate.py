from PIL import Image, ImageDraw, ImageFont
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
    #prompts match the prompts given during LoRA fine-tuning.
    style_prompts = {
        "studio ghibli": "high quality image in ghibli style, detailed, artistic",
        "LEGO": "high quality image in lego style, detailed, artistic",
        "2D animation": "high quality image in 2d_animation style, detailed, artistic",
        "3D animation": "high quality image in 3d_animation style, detailed, artistic",
    }
    
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


"""
    Converts a latent tensor to a channel overlay.
    Each channel is assigned a distinct color.
    Returns image.
"""
def visualize_latent_tensor(latent_tensor):
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
    
    overlay_image = T.ToPILImage()(composite.cpu())
    
    output_size=(192, 192)
    overlay_image = overlay_image.resize(output_size, Image.Resampling.LANCZOS)
    
    return overlay_image


def latent_channel_vis(vae, image, device):
    latent_tensor = get_latent_tensor(vae, image, device)
    return visualize_latent_tensor(latent_tensor)


"""
    Runs the image-to-image generation pipeline with intermediate latent capture.
    Computes actual denoising steps based on user-selected parameters to select steps to visualize.
    Returns final image along with visualizations of selected latent states for denoising generation.
"""
def generate_with_progression(pipe, image, prompt, device, strength, guidance_scale, num_inference_steps):
    # Calculate the actual number of steps that will be run by the pipeline.
    actual_num_steps = int(num_inference_steps * strength)
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
    
    # Add invisible watermark for accountability and traceability
    # This ensures AI-generated content can be identified even after manipulation
    # Increased strength to 0.2 for better robustness against attacks
    from src.watermark import add_watermark_to_image
    result = add_watermark_to_image(result, watermark_text="AI_GENERATED_FRANKAINSTEIN", strength=0.2)

    denoising_visualizations = [visualize_latent_tensor(latent) for latent in denoising_steps]
    denoising_visualizations_with_labels = list(zip(denoising_visualizations, capture_steps))

    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result, denoising_visualizations_with_labels


"""
    Creates an image to show user denoising steps
    Output will look as such, with 1, 3, 6,and 11 representing steps: 
    [1] -> [3] -> [6] -> [11]
"""
def create_labeled_denoising_sequence(denoising_steps_with_labels):
    if not denoising_steps_with_labels:
        return Image.new('RGB', (384, 240), color='grey')

    resized_steps = [(img.resize((192, 192), Image.Resampling.LANCZOS), step_num)
                     for img, step_num in denoising_steps_with_labels]

    arrow_width = 40
    image_width = 192
    image_height = 192
    label_height = 40
    total_height = image_height + label_height

    total_width = len(resized_steps) * image_width + (len(resized_steps) - 1) * arrow_width
    collage = Image.new('RGB', (total_width, total_height), color='black')
    draw = ImageDraw.Draw(collage)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, (img, step_num) in enumerate(resized_steps):
        x_offset = i * (image_width + arrow_width)
        collage.paste(img, (x_offset, 0))

        # Draw label centered below image
        label_text = f"Step {step_num}"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x_offset + (image_width - text_width) // 2
        text_y = image_height + (label_height - text_height) // 2
        draw.text((text_x, text_y), label_text, fill='white', font=font)

        # Draw arrow between images
        if i < len(resized_steps) - 1:
            arrow_start = x_offset + image_width
            arrow_mid_y = image_height // 2
            arrow_end = arrow_start + arrow_width - 10
            draw.line([(arrow_start, arrow_mid_y), (arrow_end, arrow_mid_y)], fill='white', width=3)
            draw.polygon([
                (arrow_end, arrow_mid_y),
                (arrow_end - 10, arrow_mid_y - 5),
                (arrow_end - 10, arrow_mid_y + 5)
            ], fill='white')

    return collage


#maybe?? not very visually appealing or educational at first glance
def visualize_prompt_guidance(pipe, prompt, device):

    text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    # The output is [1, 77, 768]. We'll visualize the 77x768 tensor.
    embedding_tensor = text_embeddings.squeeze(0) # Shape: [77, 768]
    
    # Normalize the tensor to a 0-1 range to be used as colors
    embedding_tensor = (embedding_tensor - embedding_tensor.min()) / (embedding_tensor.max() - embedding_tensor.min())

    heatmap_pil = tensor_to_pil(embedding_tensor)
    heatmap_pil = heatmap_pil.resize((384, 150), Image.Resampling.NEAREST)
    return heatmap_pil



