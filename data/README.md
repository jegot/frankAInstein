Everything in this folder was used to process and generate training pairs for model fine-tuning.

## caltech-101 

*** We did begin with this data but the initial training and models used were poor quality. 
*** Everything related to this dataset has been replaced.

This was the inital dataset downloaded from Kaggle
Source: https://www.kaggle.com/datasets/imbikramsaha/caltech-101


## additional-pairs-512
Additional fine-tuning for images of higher quality was added. 59 images were hand-picked based on
everyday objects, under-represented groups in previous dataset, and additional images that may
resonate within intended younger audience (toys, crayon drawings). The 59 images were processed as the same
inputs for each new training pair. An average of ~14 pairs were tossed for low quality generations, leaving
about 45 pairs per style.

Source: https://unsplash.com/

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
