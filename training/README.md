**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:

1. **Preserves the base model** - Original Stable Diffusion weights remain unchanged
2. **Adds lightweight adapters** - Small neural network layers that learn style-specific patterns
3. **Trains only adapters** - Much faster and memory-efficient than full fine-tuning
4. **Merges on-demand** - Adapters are combined with base model during inference

Initial training and LoRA-based model generation:

* Colab notebook used for training: https://colab.research.google.com/drive/1GOe9dIUfXwxb-CJhTEbibJwAwDT8QgRV?usp=sharing
* local copy also at training/notebooks/training.ipynb

* Training info
    - Dataset used: data/datasets/initial-caltech101-dataset
    - 50 input/output pairs per model
    - Learning rate: 1e-4 || Epochs: 5
    - ghibli: Final loss = 0.2533
    - lego: Final loss = 0.3550
    - 2d_animation: Final loss = 0.4898
    - 3d_animation: Final loss = 0.4081


Continued training for quality adjustments:

* Colab notebook used for both: https://colab.research.google.com/drive/1q2HZ-sIHGwUAhJfp_zyw_2GcbxP4T5DD
* local copy also available at training/notebooks/continued-training.ipynb

* First round info
    - Dataset used: data/datasets/additional-dataset
    - 45~ input/output pairs per model
    - Learning rate: 5e-5 || Epochs: 5
    - ghibli: Final loss = 0.0916
    - lego: Final loss = 0.1126
    - 2d_animation: Final loss = 0.1057
    - 3d_animation: Final loss = 0.1360

* Second round
    - Dataset used: data/datasets/additional-additional-dataset
    - 30~ input/output pairs per model
    - *Learning rate and epochs adjusted per style
    - ghibli: Final loss = 0.0932 (ran @ 5 epochs and lr=5e-5)
    - lego: Final loss = 0.0834 (ran @ 3 epochs and lr=1e-5)
    - 2d_animation: Final loss = 0.0854 (ran @ 3 epochs and lr=1e-5)
    - 3d_animation: Final loss = 0.0893 (ran @ 5 epochs and lr=5e-5)
