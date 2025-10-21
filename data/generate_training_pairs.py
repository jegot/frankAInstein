import os
from huggingface_hub import InferenceClient
from PIL import Image
import io
from requests import get


client = InferenceClient(
    provider="fal-ai",
    api_key="",  # replace with real huggingface token
)

image_path = r'data\additional-additional-dataset\2danimation-pairs\input'
output_path = r'data\additional-additional-datase\2danimation-pairs\output'

image_files = sorted([f for f in os.listdir(image_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))])

print(f"Found {len(image_files)} images to process")


# takes a folder and uses HuggingFace Inference Client to generate new output in target style
def process_image_with_inference(input_image_path, output_path, subject):
    
    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Skipping {output_path} (already exists)")
        return False

    with open(input_image_path, "rb") as image_file:
        input_image = image_file.read()

    prompt = f"Turn this image of a {subject} into the American_Cartoon style."

    try:
        response = client.image_to_image(
            input_image,
            prompt=prompt,
            model="Kontext-Style/American_Cartoon_lora",
        )
            # generative models used, all from huggingface.co/...:
            # "Kontext-Style/LEGO_lora"
            # "Kontext-Style/Ghibli_lora"
            # "Kontext-Style/3D_Chibi_lora"
            # "Kontext-Style/American_Cartoon_lora"

        # Case 1: Direct PIL image returned
        if isinstance(response, Image.Image):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            response.save(output_path)
            return True

        # Case 2: Dictionary with image URL
        output_data = response if isinstance(response, dict) else {}
        if "images" in output_data and output_data["images"]:
            image_url = output_data["images"][0]["url"]
            image = Image.open(get(image_url, stream=True).raw)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            return True
        else:
            print(f"No image returned for {input_image_path}. Full response: {response}")
            return False

    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")
        return False

for image_file in image_files:
    input_image_path = os.path.join(image_path, image_file)
    output_image_path = os.path.join(output_path, image_file)
    subject = image_file.split('-')[0]

    success = process_image_with_inference(input_image_path, output_image_path, subject)
    if success:
        print(f"Processed: {image_file}")
    else:
        print(f"Skipped or failed: {image_file}")

