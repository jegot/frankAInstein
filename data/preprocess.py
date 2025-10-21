import os
import shutil
from PIL import Image
 
root_dir = os.path.join('data', 'additional-additional-dataset')
#output_size = (256, 256)  # Resize images to this size
output_size = (512, 512)  # training too low quality on above size. readjusting for additional training pairs

raw_dirs = [
    os.path.join('data', 'raw1'),
    os.path.join('data', 'raw2'),
    os.path.join('data', 'raw3'),
    os.path.join('data', 'raw4')
]

# resizes images and converts to RGB
def preprocess_images_in_folder(folder_path, output_size):
    print(f"Processing folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(output_size)

                new_path = os.path.join(folder_path, filename)
                img.save(new_path)
            except Exception as e:
                print(e)

# used for larger, inital caltech-101 dataset, distributed input images across 4 style pair folders.
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
    preprocess_images_in_folder(root_dir, output_size)
    #distribute_images_to_raw_folders(root_dir, raw_dirs)