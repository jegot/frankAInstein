import torchvision.transforms as T
import gradio as gr
from PIL import Image
from src.model import load_models
from src.generate import preprocess_image, add_noise_to_image, check_nsfw_content, create_side_by_side, prompt_conversion, tensor_to_pil, latent_channel_vis, generate_with_progression, create_denoising_collage


# Global variables
pipe = None
device = None
vae = None

DEFAULT_IMAGES = None

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


def load_default_images():
    global DEFAULT_IMAGES
    asset_paths = [
        "assets/s0_cat.png",
        "assets/s1_cat.png", 
        "assets/s2_cat.png",
        "assets/s3_cat.png",
        "assets/s4_cat.png",
        "assets/s5_cat.png",
        "assets/s6_ba_cat.png"
    ]
    DEFAULT_IMAGES = [Image.open(path) for path in asset_paths]

'''
def load_map_images():
    """Load static map images"""
    try:
        map_paths = [f"assets/map{i}.png" for i in range(7)]
        return [Image.open(path) for path in map_paths]
    except Exception as e:
        print(f"Error loading map images: {e}")
'''

models_loaded = load_models_at_startup()
load_default_images()
#map_images = load_map_images()


def process_image_with_story(image, style, strength, guidance_scale, steps=20, progress=gr.Progress()):
    if image is None:
        return None, None, None, None, None, None, None, "Please upload an image first!"
    
    if not models_loaded or pipe is None:
        return None, None, None, None, None, None, None, "Models not loaded. Please refresh the page."
    
    try:
        #Preprocess 
        s0_preprocess = preprocess_image(image)

        #Encode image - pass through VAE
        s1_encode = latent_channel_vis(vae, s0_preprocess, device)

        #Initial noise injection - just a single step in SD --noise added will be based on strength chosen by user
        s2_addnoise = add_noise_to_image(s1_encode, strength)

        #***This is where things get out of order. Please note the 's#' prefixes before generated image to understand step
        s0_preprocess = tensor_to_pil(s0_preprocess)

        progress(0.5, desc="Artist is creating your new image via diffusion")
        prompt = prompt_conversion(style)


        s5_result, denoising_steps = generate_with_progression(pipe, s0_preprocess, prompt, device, strength, guidance_scale, steps)

        if check_nsfw_content(s5_result):
                    return *DEFAULT_IMAGES, "Inappropriate content detected. Default images being returned."

        #Encoded version of the final image
        s4_encodef = latent_channel_vis(vae, s5_result, device)

        progress(0.8, desc="Finishing up...")
        s3_denoise = create_denoising_collage(denoising_steps) 

        # Final step
        progress(1.0, desc="Almost ready...")

        before_after_comparison = create_side_by_side(s0_preprocess, s5_result)

        return s0_preprocess, s1_encode, s2_addnoise, s3_denoise, s4_encodef, s5_result, before_after_comparison, "Your image has been transformed!"

    except Exception as e:
        if "out of memory" in str(e).lower():
            return *DEFAULT_IMAGES, "Not enough memory to generate image. Default images being returned."

        return None, None, None, None, None, None, None, f"Error: {str(e)}"

def load_raw(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f"""{f.read()}"""

css = load_raw("theme2.css")

def create_interface():
    with gr.Blocks(
        title="frank-AI-nstein",
        theme=gr.themes.Soft(), 
        css = css
    ) as interface:
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
                
                with gr.Accordion("Advanced Settings", open=True):
                    strength = gr.Slider(0.3, 1.0, value=0.6, step=0.1, label="Style Strength", info="How much the model changes your photo & how much noise will injected")
                    guidance_scale = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Guidance Scale", info="How closely the model follows the style")

                generate_btn = gr.Button("Start the diffusion process", variant="primary", size="lg")
                status = gr.Textbox(label="Status", value="Upload an image first!", interactive=False)

            # Right Panel
            with gr.Column(scale=2):
                with gr.Tabs() as step_tabs:
                    with gr.Tab("Step 0: Original Image", id=0):
                        gr.Markdown("**Step 0:** Your image might look a little different here. It's been slightly cropped and resized to better fit the pipeline.")
                        s0_preprocess = gr.Image(label="Original Image", height=300)

                    with gr.Tab("Step 1: Latent Space Encoding", id=1):
                        gr.Markdown("**Step 1:** Your image now looks like a blurry blob. It's been compressed into 'latent space', a compact representation used by the model.")
                        s1_encode = gr.Image(label="Encoded Latent Space", height=300, show_download_button=True)

                    with gr.Tab("Step 2: Noise Injection", id=2):
                        gr.Markdown("**Step 2:** Random noise is added to the latent representation. This helps the model learn how to reconstruct and stylize your image.")
                        s2_addnoise = gr.Image(label="Noisy Image", height=300, show_download_button=True)

                    with gr.Tab("Step 3: Denoising with U-Net", id=3):
                        gr.Markdown("**Step 3:** The U-Net takes your noisy latent image and begins denoising it.")
                        s3_denoise = gr.Image(label="Reconstruction", height=300, show_download_button=True)

                    with gr.Tab("Step 4: Ready for Decoding!", id=4):
                        gr.Markdown("**Step 4:** Your new image is almost done, but it's still in latent format...")
                        s4_encodef = gr.Image(label="Your New Image", height=300, show_download_button=True)

                    with gr.Tab("Step 5: Final Result", id=5):
                        gr.Markdown("**Step 5:** Here's your AI-generated image, fully reconstructed and stylized based on your original input.")
                        s5_result = gr.Image(label="Your New Image", height=300, show_download_button=True)

                    with gr.Tab("Step 6: Before & After Comparison", id=6):
                        gr.Markdown("**Step 5:** See how your original image (left) compares to the AI-generated result (right).")
                        before_after_comparison = gr.Image(label="Before & After Comparison", height=300, show_download_button=True)

                # Navigation Buttons
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous Step")
                    next_btn = gr.Button("Next Step ➡️")

                #added 'map' section, as the user moves through result tabs above, this section will be as well with static "you are here" imgaes on a map of the diffusion pipeline
                # Map section with different variable names
                gr.Markdown("### Pipeline Map")
                with gr.Tabs() as map_tabs:
                    for i in range(7):
                        with gr.Tab(f"Step {i} Map", id=i):
                            step_names = [
                                "Original Image", "Latent Encoding", "Noise Injection", 
                                "Denoising", "Ready for Decoding", "Final Result", "Comparison"
                            ]
                            gr.Markdown(f"**You are here:** {step_names[i]}")
                            gr.Image(value=DEFAULT_IMAGES[i], height=200, show_download_button=False)


         # Fixed navigation functions
        def go_prev(current_index):
            new_index = max(current_index - 1, 0)
            return new_index, gr.Tabs(selected=new_index), gr.Tabs(selected=new_index)

        def go_next(current_index):
            new_index = min(current_index + 1, 6)
            return new_index, gr.Tabs(selected=new_index), gr.Tabs(selected=new_index)

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

        # Connect function
        generate_btn.click(
            fn=process_image_with_story,
            inputs=[input_image, style, strength, guidance_scale], #steps now default to 20 w/o user input. reason: speed increase and consistent steps in diffusion
            outputs=[
                s0_preprocess, s1_encode, s2_addnoise, s3_denoise, s4_encodef, 
                s5_result, before_after_comparison, status
            ]
        )

    return interface

#launch
if __name__ == "__main__":
    print("Starting frank-AI-nstein")
    demo = create_interface()
    demo.launch(
        debug=True,
        show_error=True
    )
    