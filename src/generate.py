from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from .model import load_models

'''
GUI GOES HERE
USER SHOULD BE ABLE TO:
- UPLOAD THEIR OWN IMAGE (FOR DEMO LETS USE A PICTURE OF ROCKY. PROFESSORS LOVE STUFF LIKE THAT)
- THE CODE BELOW WORKS FOR FILE INPUT BUT IF IT COULD LOOK MORE VISUALLY APPEALING OR BE WRAPPED WITHIN A GUI, THATD BE GOOD
- CHOOSE A STYLE FROM A PREDEFINED STYLE LIST: THINK ANIME, CARTOON, VAN GOGH...ETC. 
- SAVE THE CHOSEN STYLE AS THE 'PROMPT'

- IF POSSIBLE, ADD A FRAME IN THE GUI FOR IMAGES. TRY AND WRITE THIS AS A WHILE LOOP SO THE KIDS JUST SEE SOME IMAGES WHILE THE ACTUAL PROCESS IS RUNNING

- I'M GOING TO WORK ON SOME PICTURES TO 'VISUALIZE' THE PROCESS. HERE'S THE STORY PITCH FOR AN ANALOGY OF THE PROCESS :
    - Characters: Scientist (User), Robot (VAE), Artist (Diffuser).
    - Analogies: Machine (Encoder and Decoder), Door (bottleneck and latent space)
    1. The scientist gets a picture he wants to change the style of. He tries to run straight to the artist, but his picture is too big to fit through the door.
    2. Above the door is a sign that says 'latent-space only'
    3. A robot sitting off to the side taps on a machine. The scientist hands the picture to the robot who runs it through the machine.
    4. Out of the machine comes small, blurry version of the picture. The robot hands this to the scientist who can now fit through the door into 'latent space'.
    5. Into the latent space, the scientist meets an artist who takes the new condensed picture. He slowly starts spraying it with spray paint until it is just noise.
    6. The artist turns to the scientist and asks him to select a style from the 'style wall'. 
    7. The scientist chooses his style, and the artist begins removing noise with a paint brush.
    8. Finally, a new picture is created. It is still blurry and small like the other one.
    9. The scientist goes back through the door to the robot, who switches the setting on the machine from 'encode' to 'decode'
    10. The picture goes through the machine in the opposite direct now, and out pops the new picture in the scientist's preferred style.

    
'''

def preprocess_image(image):
    """Load and preprocess image"""
    transform = T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])  # VAE expects [-1, 1]
    ])
    x = transform(image).unsqueeze(0)  # shape: [1, 3, 512, 512]
    return x

def encode_image(vae, x, device):
    """Encode image to latent space"""
    with torch.no_grad():
        latent = vae.encode(x.to(device)).latent_dist.sample() * 0.18215  # scale factor used in SD
        reconstructed = vae.decode(latent).sample

    # Convert back to image
    reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
    reconstructed_image = T.ToPILImage()(reconstructed.squeeze().cpu())

    # Resize it to mimic latent space enconding
    new_size = (256, 256)  # diffusion model typically expects 512x512, but we'll go smaller for visualization purposes 
    reconstructed_image = reconstructed_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return reconstructed_image, latent

def add_noise_to_image(image):
    """Convert to numpy and add noise"""
    img_array = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0, 0.3, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 1)
    noisy_img = Image.fromarray((noisy_array * 255).astype(np.uint8))
    return noisy_img

def generate_style_transfer(pipe, image, prompt, device):
    """Generate style transfer using the diffusion pipeline"""
    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.5,
        guidance_scale=5,
        num_inference_steps=25  # if we make this a variable we could somehow visualize this in correlational with how many noise 'iterations' it goes through
    ).images[0]
    
    # BEFORE WE SHOW THIS TO THE USER, WE NEED TO SHOW ANOTHER LATENT TRANSFORMATION. THE IMAGE PRIOR TO THE FINAL 'DECODED ONE'.
    result.save("final_generation.png")
    
    return result