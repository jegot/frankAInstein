import os
import shutil
from PIL import Image
 
 #this file just splits up caltech-101 dataset (from Kaggle)
 #into 4 subfolder that will be used to create stylized pairs 
 #for training

root_dir = os.path.join('data', 'caltech-101')
output_size = (256, 256)  # Resize images to this size

raw_dirs = [
    os.path.join('data', 'raw1'),
    os.path.join('data', 'raw2'),
    os.path.join('data', 'raw3'),
    os.path.join('data', 'raw4')
]

def preprocess_and_rename_images(root_dir, output_size):
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")
            index = 1
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img = img.resize(output_size)

                        new_filename = f"{subfolder}{index}.jpg"
                        new_path = os.path.join(subfolder_path, new_filename)
                        img.save(new_path)

                        index += 1
                    except Exception as e:
                        print(e)


def distribute_images_to_raw_folders(root_dir, raw_dirs):
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Distributing images from: {subfolder}")
            images = sorted([
                f for f in os.listdir(subfolder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            for i, image_name in enumerate(images):
                src_path = os.path.join(subfolder_path, image_name)
                target_raw = raw_dirs[i % len(raw_dirs)]
                dst_path = os.path.join(target_raw, image_name)
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(e)

if __name__ == "__main__":
    preprocess_and_rename_images(root_dir, output_size)
    distribute_images_to_raw_folders(root_dir, raw_dirs)