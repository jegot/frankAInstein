import os
import sys
from PIL import Image
import torchvision.transforms as T

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_models
from src.generate import preprocess_image, encode_image, add_noise_to_image, generate_style_transfer


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

def main():
    """Main application function"""
    print("Please upload your image")
    
    # For non-Colab environment, we'll use a simple file input
    image_path = input("Enter the path to your image file: ")
    
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image_path}")
        
        # Display image info (in non-Colab, we can't use display() so we'll print info)
        print(f"Image size: {image.size}, Mode: {image.mode}")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Load models
    print("Loading models...")
    pipe, vae, device = load_models()
    print(f"Models loaded on device: {device}")
    
    # Preprocess image
    x = preprocess_image(image)
    
    # Encode image to latent space
    print("Encoding image to latent space...")
    reconstructed_image, latent = encode_image(vae, x, device)
    
    print("Reconstructed image (latent representation):")
    # In non-Colab, we can't display images, so we'll save them instead
    reconstructed_image.save("latent_reconstruction.png")
    print("Saved latent reconstruction as 'latent_reconstruction.png'")
    
    # Add noise
    print("Adding noise to image...")
    noisy_img = add_noise_to_image(reconstructed_image)
    noisy_img.save("noisy_image.png")
    print("Saved noisy image as 'noisy_image.png'")
    
    # Get style prompt from user
    prompt = input("Enter your desired style (e.g., 'anime style', 'van gogh style', 'cartoon style'): ")
    if not prompt:
        prompt = "anime style"  # default
    
    # Generate style transfer
    print(f"Generating image with style: {prompt}")
    result = generate_style_transfer(pipe, image, prompt, device)
    
    print("Style transfer complete!")
    print("Final result saved as 'final_generation.png'")
    
    # Display final image info
    print(f"Final image size: {result.size}")

if __name__ == "__main__":
    main()
