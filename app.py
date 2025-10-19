"""
Frank-AI-nstein: Advanced Diffusion Model with Fine-Tuned LoRA Models

This application demonstrates the complete diffusion process step-by-step,
featuring custom fine-tuned LoRA models for enhanced style transfer capabilities.
"""

import torchvision.transforms as T
import gradio as gr
import torch
from functools import partial
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL
from PIL import Image
import sys
import os

# Configure Python path to access our custom modules
sys.path.append('src')
from src.generate import preprocess_image, add_noise_to_image, check_nsfw_content, create_side_by_side, prompt_conversion, tensor_to_pil, latent_channel_vis, generate_with_progression, create_denoising_collage

# Configure path for fine-tuned model loading utilities
sys.path.append('training')
try:
    from load_finetuned import get_available_finetuned_styles, load_finetuned_model
    FINETUNED_AVAILABLE = True
except ImportError:
    print("Warning: Could not import fine-tuned model utilities")
    FINETUNED_AVAILABLE = False

# Global variables for model state management
# These are loaded lazily to improve startup time
pipe = None      # Main Stable Diffusion pipeline
device = None    # CUDA/CPU device detection
vae = None       # Variational Autoencoder for latent space operations

# Global variables for UI assets
DEFAULT_IMAGES = None  # Fallback images shown when generation fails
MAP_IMAGES = None      # Pipeline visualization images for each step

def load_default_images():
    """
    Load fallback images that are displayed when image generation fails.
    These provide a consistent user experience during error conditions.
    Each image corresponds to a specific step in the diffusion process.
    """
    global DEFAULT_IMAGES
    asset_paths = [
        "assets/s0_cat.png",  # Step 0: Original image
        "assets/s1_cat.png",  # Step 1: Latent encoding
        "assets/s2_cat.png",  # Step 2: Noise injection
        "assets/s3_cat.png",  # Step 3: Denoising process
        "assets/s4_cat.png",  # Step 4: Ready for decoding
        "assets/s5_cat.png",  # Step 5: Final result
        "assets/s6_ba_cat.png"  # Step 6: Before/after comparison
    ]
    DEFAULT_IMAGES = [Image.open(path) for path in asset_paths]

def load_map_images():
    """
    Load pipeline visualization images that show the diffusion process flowchart.
    These help users understand what happens at each step of the transformation.
    """
    global MAP_IMAGES
    asset_paths = [
        "assets/map0.gif",  # Step 0: Original image map
        "assets/map1.gif",  # Step 1: Latent encoding map
        "assets/map2.gif",  # Step 2: Noise injection map
        "assets/map3.gif",  # Step 3: Denoising map
        "assets/map4.gif",  # Step 4: Ready for decoding map
        "assets/map5.gif",  # Step 5: Final result map
        "assets/map6.png",  # Step 6: Comparison map
    ]
    MAP_IMAGES = [Image.open(path) for path in asset_paths]

def load_models():
    """
    Initialize the base Stable Diffusion model with optimal settings for the current hardware.
    Uses half-precision (float16) on GPU for memory efficiency, full precision on CPU.
    Enables attention slicing to reduce memory usage during inference.
    """
    global pipe, vae, device
    
    # Detect available compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use half-precision on GPU for memory efficiency, full precision on CPU for compatibility
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the main Stable Diffusion pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # Official Stable Diffusion 1.5 checkpoint
        torch_dtype=torch_dtype            # Optimize data type for current device
    ).to(device)
    
    # Extract VAE component for latent space operations
    vae = pipe.vae
    # Enable attention slicing to reduce memory usage during generation
    pipe.enable_attention_slicing()
    
    return pipe, vae, device

def load_finetuned_models():
    """
    Scan the training/models directory and identify available fine-tuned LoRA models.
    Returns a list of style names that have been custom trained.
    """
    if FINETUNED_AVAILABLE:
        try:
            finetuned_styles = get_available_finetuned_styles()
            print(f"Found {len(finetuned_styles)} fine-tuned models: {finetuned_styles}")
            return finetuned_styles
        except Exception as e:
            print(f"Error loading fine-tuned models: {e}")
            return []
    return []

def process_image_with_story(image, style, strength, guidance_scale, steps=20, progress=gr.Progress()):
    """
    Main image processing function that handles the complete diffusion pipeline.
    
    Args:
        image: PIL Image - Input image to transform
        style: str - Selected art style (may include "(Fine-tuned)" suffix)
        strength: float - How much to transform the image (0.3-1.0)
        guidance_scale: float - How closely to follow the style prompt (1.0-20.0)
        steps: int - Number of diffusion steps (default 20)
        progress: gr.Progress - Gradio progress tracker for user feedback
    
    Returns:
        tuple: 7 processed images + status message for display in UI tabs
    """
    global pipe, vae, device
    
    # Lazy load models - only initialize when first image is processed
    if pipe is None:
        progress(0.1, desc="Loading models...")
        pipe, vae, device = load_models()
    
    if image is None:
        return None, None, None, None, None, None, None, "Please upload an image first!"
    
    try:
        current_device = device
        
        # Check if user selected a fine-tuned style and load appropriate model
        is_finetuned = False
        style_name = None
        
        if " (Fine-tuned)" in style:
            is_finetuned = True
            style_name = style.replace(" (Fine-tuned)", "")
            progress(0.15, desc=f"Loading fine-tuned {style_name} model...")
            # Load and merge LoRA adapter with base model
            pipe = load_finetuned_model(pipe, style_name)
        
        # Step 0: Preprocess input image and convert style to prompt
        prompt = prompt_conversion(style.replace(" (Fine-tuned)", ""))
        s0_preprocess = preprocess_image(image)  # Convert to tensor, resize, normalize

        # Step 1: Encode image into latent space using VAE
        s1_encode = latent_channel_vis(vae, s0_preprocess, current_device)

        # Step 2: Add noise based on strength parameter for visualization
        s2_addnoise = add_noise_to_image(s1_encode, strength)

        # Convert preprocessed tensor back to PIL for display
        s0_preprocess = tensor_to_pil(s0_preprocess)

        progress(0.5, desc="Generating image...")
        
        # Steps 3-5: Main diffusion process with denoising
        s5_result, denoising_steps = generate_with_progression(
            pipe, s0_preprocess, prompt, current_device, 
            strength, guidance_scale, steps
        )

        # Safety check for inappropriate content
        if check_nsfw_content(s5_result):
            return *DEFAULT_IMAGES, "Inappropriate content detected. Default images being returned."

        # Step 4: Encode final result back to latent space for visualization
        s4_encodef = latent_channel_vis(vae, s5_result, current_device)

        progress(0.8, desc="Finishing up...")
        
        # Step 3: Create denoising progression collage
        s3_denoise = create_denoising_collage(denoising_steps) 

        progress(1.0, desc="Almost ready...")

        # Step 6: Create before/after comparison
        before_after_comparison = create_side_by_side(s0_preprocess, s5_result)

        return s0_preprocess, s1_encode, s2_addnoise, s3_denoise, s4_encodef, s5_result, before_after_comparison, "Image transformation complete!"

    except Exception as e:
        # Handle out of memory errors gracefully
        if "out of memory" in str(e).lower():
            return *DEFAULT_IMAGES, "Not enough memory to generate image. Default images being returned."

        return None, None, None, None, None, None, None, f"Error: {str(e)}"

def load_raw(filepath):
    """
    Utility function to load text files (like CSS) with proper encoding.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f"""{f.read()}"""

def create_interface():
    """
    Create the main Gradio interface with fine-tuned model integration.
    Builds a sophisticated UI with step-by-step visualization and model selection.
    """
    # Get available fine-tuned styles at startup
    finetuned_styles = load_finetuned_models()
    
    # Define base model styles available without fine-tuning
    base_styles = [
        "anime style", "van gogh style", "watercolor painting", "pixel art", 
        "3D animation", "cyberpunk style"
    ]
    
    # Add fine-tuned styles with special labeling for user clarity
    finetuned_options = [f"{style} (Fine-tuned)" for style in finetuned_styles]
    
    # Combine styles, prioritizing fine-tuned ones for better user experience
    all_styles = finetuned_options + base_styles
    
    # Default to first fine-tuned style if available, otherwise first base style
    default_style = finetuned_options[0] if finetuned_options else base_styles[0]
    
    # Load custom CSS theme if available
    try:
        css = load_raw("src/theme.css")
    except:
        css = ""

    # Create the main Gradio interface with custom theme
    with gr.Blocks(
        title="frank-AI-nstein",
        theme=gr.themes.Default(),  # Use default theme as base
        css=css                     # Apply our custom dark green sci-fi theme
    ) as interface:
        
        # Header section with dynamic fine-tuned model count
        gr.HTML(f"""
        <div class="main-header">
            <h1> frank-AI-nstein</h1>
            <p> Learn how diffusion models transform images! Watch the process happen step by step.</p>
            <p><strong>🎨 {len(finetuned_styles)} Fine-tuned Models Available!</strong></p>
        </div>
        """)
        
        # State variable to track current tab for navigation
        tab_index = gr.State(value=0)

        with gr.Row():
            # Left Panel - Controls and Settings
            with gr.Column(scale=1, elem_classes="step-card primary-card"):
                gr.Markdown("## Get Started")
                
                # Image upload component with custom styling
                input_image = gr.Image(
                    label="Upload Your Photo", 
                    type="pil", 
                    height=250, 
                    show_download_button=False, 
                    elem_classes="gradio-image"
                )

                # Style selection dropdown with fine-tuned options
                style = gr.Dropdown(
                    choices=all_styles,
                    value=default_style,
                    label="Choose Your New Art Style",
                    info="Fine-tuned models (marked) provide enhanced results",
                    elem_classes="gradio-dropdown"    
                )
                
                # Advanced settings in collapsible accordion
                with gr.Accordion("Advanced Settings", open=True):
                    strength = gr.Slider(
                        0.3, 1.0, value=0.6, step=0.1, 
                        label="Style Strength", 
                        info="How much the model changes your photo & how much noise will injected"
                    )
                    guidance_scale = gr.Slider(
                        1.0, 20.0, value=5.0, step=0.5, 
                        label="Guidance Scale", 
                        info="How closely the model follows the style"
                    )

                # Main action button with custom styling and animations
                generate_btn = gr.Button(
                    "Start the diffusion process", 
                    variant="primary", 
                    size="lg", 
                    elem_classes="generate-btn"
                )
                
                # Status display with custom styling
                status = gr.Textbox(
                    label="Status", 
                    value="Upload an image first!", 
                    interactive=False, 
                    elem_classes="status-text"
                )

            # Right Panel - Step-by-step Visualization
            with gr.Column(scale=2):
                with gr.Tabs() as step_tabs:
                    # Step 0: Original Image
                    with gr.Tab("Step 0: Original Image", id=0):
                        gr.Markdown("**Step 0:** Your image might look a little different here. It's been slightly cropped and resized to better fit the pipeline.")
                        s0_preprocess = gr.Image(label="Original Image", height=300)

                    # Step 1: Latent Space Encoding
                    with gr.Tab("Step 1: Latent Space Encoding", id=1):
                        gr.Markdown("**Step 1:** Your image now looks like a blurry blob. It's been compressed into 'latent space', a compact representation used by the model.")
                        s1_encode = gr.Image(label="Encoded Latent Space", height=300, show_download_button=True)

                    # Step 2: Noise Injection
                    with gr.Tab("Step 2: Noise Injection", id=2):
                        gr.Markdown("**Step 2:** Random noise is added to the latent representation. This helps the model learn how to reconstruct and stylize your image.")
                        s2_addnoise = gr.Image(label="Noisy Image", height=300, show_download_button=True)

                    # Step 3: Denoising with U-Net
                    with gr.Tab("Step 3: Denoising with U-Net", id=3):
                        gr.Markdown("**Step 3:** The U-Net takes your noisy latent image and begins denoising it.")
                        s3_denoise = gr.Image(label="Reconstruction", height=300, show_download_button=True)

                    # Step 4: Ready for Decoding
                    with gr.Tab("Step 4: Ready for Decoding!", id=4):
                        gr.Markdown("**Step 4:** Your new image is almost done, but it's still in latent format...")
                        s4_encodef = gr.Image(label="Your New Image", height=300, show_download_button=True)

                    # Step 5: Final Result
                    with gr.Tab("Step 5: Final Result", id=5):
                        gr.Markdown("**Step 5:** Here's your generated image, fully reconstructed and stylized based on your original input.")
                        s5_result = gr.Image(label="Your New Image", height=300, show_download_button=True)

                    # Step 6: Before & After Comparison
                    with gr.Tab("Step 6: Before & After Comparison", id=6):
                        gr.Markdown("**Step 6:** See how your original image (left) compares to the generated result (right).")
                        before_after_comparison = gr.Image(label="Before & After Comparison", height=300, show_download_button=True)

                # Navigation buttons for step-by-step exploration
                with gr.Row():
                    prev_btn = gr.Button("<- Previous Step")
                    next_btn = gr.Button("Next Step ->")

                # Pipeline map visualization section
                gr.Markdown("### Pipeline Map")
                with gr.Tabs() as map_tabs:
                    for i in range(7):
                        with gr.Tab(f"Step {i} Map", id=i):
                            step_names = [
                                "Original Image", "Latent Encoding", "Noise Injection", 
                                "Denoising", "Ready for Decoding", "Final Result", "Comparison"
                            ]
                            # Only show map if image exists
                            if i < len(MAP_IMAGES):
                                gr.Image(
                                    value=MAP_IMAGES[i], 
                                    height=300, 
                                    show_download_button=False, 
                                    type="pil", 
                                    format="gif"
                                )

        # Navigation functions for step-by-step exploration
        def go_prev(current_index):
            """Navigate to previous step, with bounds checking."""
            new_index = max(current_index - 1, 0)
            return new_index, gr.Tabs(selected=new_index), gr.Tabs(selected=new_index)

        def go_next(current_index):
            """Navigate to next step, with bounds checking."""
            new_index = min(current_index + 1, 6)
            return new_index, gr.Tabs(selected=new_index), gr.Tabs(selected=new_index)

        # Connect navigation buttons
        prev_btn.click(
            fn=go_prev,
            inputs=tab_index,
            outputs=[tab_index, step_tabs, map_tabs]
        )
        next_btn.click(
            fn=go_next,
            inputs=tab_index,
            outputs=[tab_index, step_tabs, map_tabs]
        )

        # Connect main generation function
        generate_btn.click(
            fn=process_image_with_story,
            inputs=[input_image, style, strength, guidance_scale],
            outputs=[
                s0_preprocess, s1_encode, s2_addnoise, s3_denoise, s4_encodef, 
                s5_result, before_after_comparison, status
            ]
        )

    return interface


# Main execution block
if __name__ == "__main__":
    print("Starting frank-AI-nstein")
    
    # Load UI assets (not heavy models yet - lazy loading)
    load_default_images()
    load_map_images()

    # Create and launch the interface
    demo = create_interface()
    
    # Launch with custom port and settings
    demo.queue().launch(
        server_name="127.0.0.1",  # Local access only
        server_port=73,           # Custom port to avoid conflicts
        debug=True,               # Enable debug mode
        show_error=True           # Show detailed error messages
    )
