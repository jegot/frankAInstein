import torchvision.transforms as T
import gradio as gr
from PIL import Image

from src.model import load_models
from src.generate import preprocess_image, encode_image, add_noise_to_image, generate_style_transfer, check_nsfw_content, create_side_by_side, prompt_conversion

# Global variables
pipe = None
device = None
vae = None

def load_models_at_startup():
    """Load models once at startup"""
    global pipe, device, vae
    try:
        pipe, device, vae = load_models()
        print("Models loaded, launch interface")
        return True
    
    except Exception as e:
        print("ERROR, models could not be loaded")
        return False
models_loaded = load_models_at_startup()

def process_image_with_story(image, style, strength, guidance_scale, steps, progress=gr.Progress()):
    if image is None:
        return None, None, None, None, None, "Please upload an image first!"
    
    if not models_loaded or pipe is None:
        return None, None, None, None, None, "❌ Models not loaded. Please refresh the page."
    
    try:
        # Step 0: Preprocess 
        x = preprocess_image(image)

        # Step 1: Encode image - pass through VAE
        reconstructed_image = encode_image(vae, x, device)

        # Step 2: Add noise ***UPDATE SO THAT AMOUNT OF NOISE IS VARIABLE WITH HOW MUCH USER SELECTS
        noisy_img = add_noise_to_image(reconstructed_image)

        # Step 3: Diffusion process - FIXED
        progress(0.5, desc="Artist is creating your new image via diffusion")
        prompt = prompt_conversion(style)
        result = generate_style_transfer(pipe, image, prompt, device, strength, guidance_scale, steps)
        if check_nsfw_content(result):
                    progress(0.7, desc="Innappropriate content detected. Default images being returned...")
                    result = generate_style_transfer(pipe, image, style, device, strength, guidance_scale, steps)


        # create latent version of generated image
        latent_final = encode_image(vae, result, device)

        #create noisy version of new latent
        noisy_final_latent = add_noise_to_image(latent_final) #ADD TO RETURN...NEW STEP!!!

        if check_nsfw_content(result):
            progress(0.7, desc="Regenerating: Creating a new version...")
            result = generate_style_transfer(pipe, image, style, device, strength, guidance_scale, steps)

        # Final step
        progress(1.0, desc="Almost ready...")

        before_after_comparison = create_side_by_side(image, result)

        return reconstructed_image, noisy_img, latent_final, result, before_after_comparison, "Your image has been transformed!"

    except Exception as e:
        if "out of memory" in str(e).lower():
            return None, None, None, None, None, "⚠️ Not enough memory to generate image. Try lowering the steps or using a smaller image."

        return None, None, None, None, None, f"Error: {str(e)}"

def create_interface():
    with gr.Blocks(
        title="frank-AI-nstein",
        theme=gr.themes.Citrus()
    ) as interface:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>frank-AI-nstein</h1>
            <p>Learn how diffusion models transform images! Watch the process happen step by step.</p>
        </div>
        """)
        
        tab_index = gr.State(value=0)

        with gr.Row():
            # Left Panel
            with gr.Column(scale=1, elem_classes="step-card primary-card"):
                gr.Markdown("## Get Started")
                input_image = gr.Image(label="Upload Your Photo", type="pil", height=250, show_download_button=False, elem_classes="gradio-image")

                style = gr.Dropdown(
                    choices=["anime style", "cartoon style", "van gogh style", "watercolor painting", "pixel art", 
                             "sketch drawing", "oil painting", "cyberpunk style", "vintage poster"], 
                    value="anime style", 
                    label="Choose Your New Art Style",
                    elem_classes="gradio-dropdown"    
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=True):
                    strength = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Style Strength", info="How much the AI changes your photo")
                    guidance_scale = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Guidance Scale", info="How closely AI follows the style")
                    steps = gr.Slider(10, 50, value=25, step=5, label="Generation Steps", info="More steps = higher quality (slower)")

                generate_btn = gr.Button("Start the diffusion process", variant="primary", size="lg")
                status = gr.Textbox(label="Status", value="Ready to create magic! Upload an image first", interactive=False)

                loading_indicator = gr.HTML("""
                <div id="loading-indicator" style="display: none; text-align: center; margin-top: 10px;">
                    <div class="progress-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <p style="margin-top: 10px; color: #667eea; font-weight: 600;">Generation in progress...</p>
                </div>
                """)

            # Right Panel
            with gr.Column(scale=2):
                with gr.Tabs() as step_tabs:
                    with gr.Tab("Step 0: Original Image", id=0):
                        gr.Markdown("**Step 0:** Your image might look a little different here. It's been slightly cropped and resized to better fit the pipeline.")
                        image_resized = gr.Image(label="Original Image", height=300)

                    with gr.Tab("Step 1: Latent Space Encoding", id=1):
                        gr.Markdown("**Step 1:** Your image now looks like a blurry blob. It's been compressed into 'latent space', a compact representation used by the model.")
                        compressed_image = gr.Image(label="Encoded Latent Space", height=300, show_download_button=True)

                    with gr.Tab("Step 2: Noise Injection", id=2):
                        gr.Markdown("**Step 2:** Random noise is added to the latent representation. This helps the model learn how to reconstruct and stylize your image.")
                        noisy_image = gr.Image(label="Noisy Image", height=300, show_download_button=True)

                    with gr.Tab("Step 3: Denoising with U-Net", id=3):
                        gr.Markdown("**Step 3:** The U-Net takes your noisy latent image and begins reconstructing it. It removes noise while preserving structure.")
                        latent_result = gr.Image(label="Reconstruction", height=300, show_download_button=True)

                    with gr.Tab("Step 4: Final Result", id=4):
                        gr.Markdown("**Step 4:** Here's your AI-generated image, fully reconstructed and stylized based on your original input.")
                        final_result = gr.Image(label="Your New Image", height=300, show_download_button=True)

                    with gr.Tab("Step 5: Before & After Comparison", id=5):
                        gr.Markdown("**Step 5:** See how your original image (left) compares to the AI-generated result (right).")
                        comparison = gr.Image(label="Before & After Comparison", height=300, show_download_button=True)

                # Navigation Buttons
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous Step")
                    next_btn = gr.Button("Next Step ➡️")


        def navigate_tabs(direction):
            def update_index(current_index):
                if direction == "next":
                    new_index = min(current_index + 1, 5)
                else:  # previous
                    new_index = max(current_index - 1, 0)
                return new_index, gr.Tabs(selected=new_index)
            
            return update_index

        prev_btn.click(
            fn=lambda idx: (max(idx - 1, 0), gr.Tabs(selected=max(idx - 1, 0))),
            inputs=tab_index,
            outputs=[tab_index, step_tabs]
        )

        next_btn.click(
            fn=lambda idx: (min(idx + 1, 5), gr.Tabs(selected=min(idx + 1, 5))),
            inputs=tab_index,
            outputs=[tab_index, step_tabs]
        )

        # Connect function
        generate_btn.click(
            fn=process_image_with_story,
            inputs=[input_image, style, strength, guidance_scale, steps],
            outputs=[compressed_image, noisy_image, latent_result, final_result, comparison, status]
        )

    return interface

#launch
if __name__ == "__main__":
    print("Starting frank-AI-nstein")

    
    img.save('img_name.png')

    '''
    
    demo = create_interface()
    demo.launch(
        debug=True,
        show_error=True
    )
    '''