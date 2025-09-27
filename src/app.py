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
- UPLOAD THEIR OWN IMAGE
- THE CODE BELOW WORKS FOR FILE INPUT BUT IF IT COULD LOOK MORE VISUALLY APPEALING OR BE WRAPPED WITHIN A GUI, THATD BE GOOD
- CHOOSE A STYLE FROM A PREDEFINED STYLE LIST: THINK ANIME, CARTOON, VAN GOGH...ETC. 
- SAVE THE CHOSEN STYLE AS THE 'PROMPT'

- IF POSSIBLE, ADD A FRAME IN THE GUI FOR IMAGES. TRY AND WRITE THIS AS A WHILE LOOP SO THE KIDS JUST SEE SOME IMAGES WHILE THE ACTUAL PROCESS IS RUNNING
'''

def main():
    """Main application function"""
    print("Please upload your image")
    
    
    image_path = input("Enter the path to your image file: ")
    
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image_path}")
        
        # Display image info (in non-Colab, we can't use display(), lets find an alternative)
        
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
    
    reconstructed_image.save("latent_reconstruction.png")
    
    
    # Add noise
    # if we could do this in iterations, that would be more educationally beneficial
    print("Adding noise to image...")
    noisy_img = add_noise_to_image(reconstructed_image)
    noisy_img.save("noisy_image.png")
    print("Saved noisy image as 'noisy_image.png'")


    
    # Get style prompt from user
    #adjust this so that it is limited to a static list. kids should not have direct prompt access for safety reasons
    prompt = input("Enter your desired style (e.g., 'anime style', 'van gogh style', 'cartoon style'): ")
    if not prompt:
        prompt = "anime style"  # default
    

    # BEFORE WE SHOW THIS TO THE USER, WE NEED TO SHOW ANOTHER LATENT TRANSFORMATION. THE IMAGE PRIOR TO THE FINAL 'DECODED ONE'.

 
    result = generate_style_transfer(pipe, image, prompt, device)
    print("Final result saved as 'final_generation.png'")


if __name__ == "__main__":
    main()
