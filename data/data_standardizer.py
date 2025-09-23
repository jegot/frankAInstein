# this is from a project i used before, it just goes through the images and make sure everything is the same size

import os
from PIL import Image

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, filename in enumerate(os.listdir(input_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize(size)
            new_name = f"img_{i:04d}.jpg"
            img_resized.save(os.path.join(output_dir, new_name))
            print(f"Saved: {new_name}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_images("raw/cat", "cat")