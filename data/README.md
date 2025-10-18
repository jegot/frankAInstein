Everything in this folder was used to process and generate training pairs for model fine-tuning.


## caltech-101
This was the inital dataset downloaded from Kaggle
Source: https://www.kaggle.com/datasets/imbikramsaha/caltech-101

Dataset was scaled down, leaving only about 4-8 images per object category.
Innapropriate/poor quality images were deleted. Any object categories that seemed obsolete to the target audience were also deleted entirely.

## preprocess.py
Resized (256, 256), renamed, and coverted all images to RBG in the caltech-101 dataset.
Processed images were then split between 4 styled-pair folders into their input subfolders:
    - 2Danimation-pairs
    - 3Danimation-pairs
    - ghibli-pairs
    - lego-pairs 

## generate_training_pairs
Takes the input subfolder from each of the folders above and generates outputs in the targeted style. 
All data was generated using one of Kontext-Style's models on HuggingFace.
Source: https://huggingface.co/Kontext-Style/models

*NOTE: It is important to note that we are gathering our own data for training rather than simply using the weights from 
the model above for speed, size, and compatibility purposes.  
