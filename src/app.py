import os
import sys
from PIL import Image
import torchvision.transforms as T

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_models
from src.generate import preprocess_image, encode_image, add_noise_to_image, generate_style_transfer

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
