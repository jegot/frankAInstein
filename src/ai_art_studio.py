# Import our existing modular structure
from .model import load_models
from .generate import preprocess_image, encode_image, add_noise_to_image, generate_style_transfer
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import ipywidgets as widgets
from diffusers.models import AutoencoderKL
from sklearn.manifold import TSNE
import torchvision.transforms as T
import gradio as gr
import time

# Global variables for models (loaded once)
pipe = None
vae = None
device = None

def load_models_once():
    """Load models once at startup using our existing model.py"""
    global pipe, vae, device
    
    if pipe is None:  # Only load once
        pipe, vae, device = load_models()
        print(f"Models loaded on device: {device}")
    
    return pipe, vae, device


def check_nsfw_content(image):
    """Check if image is all black (NSFW filter triggered)"""
    img_array = np.array(image)
    # Check if all pixels are black (or very close to black)
    black_pixels = np.sum(img_array < 10)  # Count pixels that are very dark
    total_pixels = img_array.size
    black_ratio = black_pixels / total_pixels

    # If more than 95% of pixels are black, it's likely NSFW filtered
    return black_ratio > 0.95


def process_image_with_story(image, style, strength, guidance_scale, steps, progress=gr.Progress()):

    if image is None:
        return None, None, None, None, None, None, "Please upload an image first!"

    try:
        # Load models using our existing modular structure
        pipe, vae, device = load_models_once()

        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

        # Step 1: GUI output
        progress(0.1, desc="Encoding your image...")
        time.sleep(1)  # Longer delay for fun effect

        # Preprocess image using our existing function
        x = preprocess_image(image)
        
        # Encode image using our existing function
        reconstructed_image, latent = encode_image(vae, x, device)




        # Step 2: Noise added to mimic internal diffusion process
        progress(0.3, desc="Adding noise for diffusion")
        time.sleep(1)  # Fun delay

        # Add noise using our existing function
        noisy_img = add_noise_to_image(reconstructed_image)


        # Step 3: Diffusion process
        progress(0.5, desc="Artist: Working its magic...")
        time.sleep(1)  # delay

        # Generate final image using our existing function
        result = generate_style_transfer(pipe, image, style, device, strength, guidance_scale, steps)


        # Create a different image for "Artist Creates Style" step - show the latent processing
        with torch.no_grad():
            # Show what the artist is working on in latent space
            result_tensor = transform(result).unsqueeze(0)
            result_latent = vae.encode(result_tensor.to(device)).latent_dist.sample() * 0.18215
            # Decode it to show the artist's work
            artist_work = vae.decode(result_latent).sample
            artist_work = (artist_work / 2 + 0.5).clamp(0, 1)
            artist_work_image = T.ToPILImage()(artist_work.squeeze().cpu())
            artist_work_image = artist_work_image.resize((256, 256), Image.Resampling.LANCZOS)



        # NSFW Checker - Check if result is all black (NSFW filtered)
        progress(0.6, desc="Safety Check")
        time.sleep(0.5)

        if check_nsfw_content(result):
            progress(0.6, desc="Regenerating: Creating a new version...")
            time.sleep(1)
            # Regenerate with different seed if NSFW detected
            result = pipe(
                prompt=style,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device=device).manual_seed(123),  # Different seed
                eta=0.0,
                output_type="pil"
            ).images[0]


        # Step 4: Show latent transformation of final result
        progress(0.7, desc="🔬 Final Check: Seeing how the AI sees your image...")
        time.sleep(1)


        # Encode the final result back to latent space to show how it looks
        with torch.no_grad():
            result_tensor = transform(result).unsqueeze(0)
            result_latent = vae.encode(result_tensor.to(device)).latent_dist.sample() * 0.18215
            result_reconstructed = vae.decode(result_latent).sample
            result_reconstructed = (result_reconstructed / 2 + 0.5).clamp(0, 1)
            result_latent_image = T.ToPILImage()(result_reconstructed.squeeze().cpu())
            result_latent_image = result_latent_image.resize((512, 512), Image.Resampling.LANCZOS)



        # Final celebration
        progress(1.0, desc="🎉 AI Magic Complete! Your masterpiece is ready!")
        time.sleep(0.5)

        # Create side-by-side before/after comparison
        def create_side_by_side(before_img, after_img):
            """Create a side-by-side comparison image"""
            # Resize both images to same height
            target_height = 512
            before_resized = before_img.resize((target_height, target_height), Image.Resampling.LANCZOS)
            after_resized = after_img.resize((target_height, target_height), Image.Resampling.LANCZOS)

            # Create side-by-side image
            comparison_width = target_height * 2
            comparison_img = Image.new('RGB', (comparison_width, target_height))
            comparison_img.paste(before_resized, (0, 0))
            comparison_img.paste(after_resized, (target_height, 0))

            return comparison_img


        # Create the before/after comparison
        before_after_comparison = create_side_by_side(image, result)


        # Debug: Print image sizes
        print(f"Debug - Reconstructed image size: {reconstructed_image.size}")
        print(f"Debug - Noisy image size: {noisy_img.size}")
        print(f"Debug - Artist work image size: {artist_work_image.size}")
        print(f"Debug - Result image size: {result.size}")
        print(f"Debug - Final latent image size: {result_latent_image.size}")
        print(f"Debug - Comparison image size: {before_after_comparison.size}")

        return reconstructed_image, noisy_img, artist_work_image, result, before_after_comparison, "Your image has been transformed! 🎨✨"

    except Exception as e:
        return None, None, None, None, None, None, f"❌ Error: {str(e)}"



def create_vercel_style_interface():

    with gr.Blocks(
        title="FrankAInstein - AI Art Magic Studio",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate"
        ),
        css="../styles/vercel_theme.css"

    ) as interface:

        # Vercel-style Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 FrankAInstein</h1>
            <h2>AI Art Magic Studio</h2>
            <p>Learn how AI creates amazing art! Watch the magic happen step by step.</p>
        </div>
        """)

        with gr.Row():
            # Left Panel - Upload & Controls (Vercel-style)
            with gr.Column(scale=1, elem_classes="step-card primary-card"):
                gr.Markdown("## 🎯 Get Started")

                # Image Upload
                input_image = gr.Image(
                    label="📸 Upload Your Photo",
                    type="pil",
                    height=250,
                    show_download_button=False,
                    elem_classes="gradio-image"
                )

                # Style Selection
                style = gr.Dropdown(
                    choices=[
                        "anime style",
                        "cartoon style",
                        "van gogh style",
                        "watercolor painting",
                        "pixel art",
                        "sketch drawing",
                        "oil painting",
                        "digital art",
                        "cyberpunk style",
                        "vintage poster"
                    ],
                    value="anime style",
                    label="🎨 Choose Your Art Style",
                    info="Pick a fun style for your image!",
                    elem_classes="gradio-dropdown"
                )

                # Advanced Controls
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Style Strength",
                        info="How much the AI changes your photo",
                        elem_classes="gradio-slider"
                    )

                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=5.0,
                        step=0.5,
                        label="Guidance Scale",
                        info="How closely AI follows the style",
                        elem_classes="gradio-slider"
                    )

                    steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=5,
                        label="Generation Steps",
                        info="More steps = higher quality (slower)",
                        elem_classes="gradio-slider"
                    )

                # Generate Button
                generate_btn = gr.Button(
                    "🚀 Start the AI Magic!",
                    variant="primary",
                    size="lg",
                    elem_classes="generate-btn"
                )

                # Status with fun loading animation
                status = gr.Textbox(
                    label="Status",
                    value="Ready to create magic! Upload an image and click 'Start the AI Magic!'",
                    interactive=False,
                    max_lines=3,
                    elem_classes="status-text"
                )

                # Fun loading indicator
                loading_indicator = gr.HTML("""
                <div id="loading-indicator" style="display: none; text-align: center; margin-top: 10px;">
                    <div class="progress-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <p style="margin-top: 10px; color: #667eea; font-weight: 600;">AI is working its magic...</p>
                </div>
                """)

            # Right Panel - Results (Vercel-style)
            with gr.Column(scale=2):
                gr.Markdown("## 🎭 Watch the AI Magic Happen!")

                with gr.Tabs(elem_classes="gradio-tabs"):
                    with gr.Tab("🤖 Step 1: Robot Compresses", elem_classes="gradio-tab"):
                        gr.Markdown("**The Robot Helper compresses your image so it can fit through the AI door!**")
                        compressed_image = gr.Image(
                            label="Compressed Image (Latent Space)",
                            height=300,
                            show_download_button=True
                        )

                    with gr.Tab("🎨 Step 2: Artist Adds Noise", elem_classes="gradio-tab"):
                        gr.Markdown("**The Artist sprays colorful paint (noise) on your compressed image!**")
                        noisy_image = gr.Image(
                            label="Noisy Image",
                            height=300,
                            show_download_button=True
                        )

                    with gr.Tab("✨ Step 3: Artist Creates Style", elem_classes="gradio-tab"):
                        gr.Markdown("**The Artist works in the latent space to create your new style!**")
                        latent_result = gr.Image(
                            label="Artist's Work in Progress",
                            height=300,
                            show_download_button=True
                        )

                    with gr.Tab("🔬 Step 4: Final Result", elem_classes="gradio-tab"):
                        gr.Markdown("**Your beautifully styled image is ready!**")
                        final_result = gr.Image(
                            label="Your New Styled Image",
                            height=300,
                            show_download_button=True
                        )

                    with gr.Tab("📊 Before & After", elem_classes="gradio-tab"):
                        gr.Markdown("**Compare your original image (left) with the AI-generated result (right)!**")
                        comparison = gr.Image(
                            label="Before & After Comparison (Original → Styled)",
                            height=300,
                            show_download_button=True
                        )

        with gr.Row(elem_classes="story-section"):
            gr.Markdown("""
            ## 🧠 How AI Image Generation Works

            **The Educational Story:**

            1. **🤖 Robot Helper (VAE Encoder)**: Your image is too big for the AI processing room, so the robot compresses it into a smaller, blurry version that can fit through the "latent space door"

            2. **🎨 Artist (Diffusion Model)**: The artist takes your compressed image and sprays it with colorful paint (noise) until it becomes just random colors

            3. **✨ Magic Process**: The artist then carefully removes the noise while applying your chosen style, creating a new compressed image

            4. **🔍 Safety Check**: The AI checks if the image is appropriate and regenerates if needed

            5. **🔬 Final Result**: Your beautifully styled image is ready!

            **This is how AI learns to create images - by understanding how to remove noise while adding artistic style!**
            """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="character-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">👤</div>
                    <h4 style="margin: 0; font-weight: 600;">You</h4>
                    <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">The Creative Human</p>
                </div>
                """)

            with gr.Column():
                gr.HTML("""
                <div class="character-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                    <h4 style="margin: 0; font-weight: 600;">Robot Helper</h4>
                    <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">VAE Encoder/Decoder</p>
                </div>
                """)

            with gr.Column():
                gr.HTML("""
                <div class="character-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎨</div>
                    <h4 style="margin: 0; font-weight: 600;">AI Artist</h4>
                    <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">Diffusion Model</p>
                </div>
                """)

        # Connect the function to the interface
        generate_btn.click(
            fn=process_image_with_story,
            inputs=[input_image, style, strength, guidance_scale, steps],
            outputs=[compressed_image, noisy_image, latent_result, final_result, comparison, status]
        )

    return interface

print("🚀 Starting FrankAInstein - AI Art Magic Studio...")

# Create and launch the interface
interface = create_vercel_style_interface()
interface.launch(
    share=True,  # Creates a public link
    debug=True,
    show_error=True
)
