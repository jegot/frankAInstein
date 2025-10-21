### Datasets

## initial-caltech-101 

Original dataset of 9,000+, sized down significantly based on image quality and usage.
About 500 images were distributed across 4 different folders for initial LoRA training. Results from
training on this dataset yielded poor performance, likely from reduced image sizes (256x56)

Source: https://www.kaggle.com/datasets/imbikramsaha/caltech-101

## additional
Additional fine-tuning for images of higher quality was added for continued model training. 59 images were hand-picked 
based on everyday objects, under-represented groups in previous dataset, and additional images that may
resonate within intended younger audience (toys, crayon drawings). The 59 images were processed as the same
inputs for each new training pair. An average of ~14 pairs were tossed for low quality generations, leaving
about 45 pairs per style.

Source: https://unsplash.com/

## additional-additional
As the name suggests, this was for the second round of continued learning for each LoRA. It uses preprocessing
logic as the other datasets. ~30 pairs were generated.

Source: https://unsplash.com/


### Processing files

## preprocess.py
Resizes (256 initially, bumped up to 512 for quality) and coverts all images to RBG in the dataset.

## generate_training_pairs
Takes the input subfolder from each of the folders above and generates outputs in the targeted style. 
All data was generated using one of Kontext-Style's models on HuggingFace.
Source: https://huggingface.co/Kontext-Style/models

*NOTE: It is important to note that we are gathering our own data for training rather than simply using the weights from 
the model above for speed, size, and compatibility purposes.  
