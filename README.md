# frankAInstein

An educational AI image generation application that demonstrates how Stable Diffusion works through interactive visualization. Designed to teach children and beginners about the process of AI image generation.

## Overview

frankAInstein is a comprehensive educational tool that breaks down the complex process of AI image generation into understandable, visual steps. The application uses Stable Diffusion combined with fine-tuned LoRA adapters to transform user images into different styles while providing real-time visualization of each step in the process.

## Educational Purpose

This project aims to make AI image generation accessible and understandable to young learners by:

* Visualizing each step of the diffusion process
* Providing interactive demonstrations of latent space encoding
* Showing how noise addition and removal works
* Demonstrating the complete image transformation pipeline

### Core Components

**Model Management (`src/model.py`)**
* Loads Stable Diffusion v1.5 as an overall generative base model
* VAE (Variational Autoencoder) from Stable Diffusion v1.4 loaded specficially for encoding/decoding visualizations
* Provides efficient model loading with one-time initialization

**Image Processing (`src/generate.py`)**
* Preprocesses images for AI model input (resize, normalize, tensor conversion)
* Handles VAE encoding and decoding for latent space visualization
* Manages noise addition for diffusion process demonstration
* Executes the main style transfer generation with configurable parameters

**User Interfaces**
* (`app.py`)**: Gradio-based web application with visual progress tracking

## Installation

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU is ideal, but running on CPU is managable. 
* 8GB+ RAM (16GB recommended)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/jegot/frankAInstein.git
cd frankAInstein
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python app.py
```

## Usage

Launch the interactive web application:
```bash
python app.py
```

The web interface provides:
* Drag and drop image upload
* Style selection from predefined options
* Real-time progress visualization
* Step-by-step process demonstration
* Before and after image comparison
* Downloadable results


## Technical Details

### Models Used

* **Primary Generation**: `runwayml/stable-diffusion-v1-5`
  * Handles the main image-to-image diffusion process
  * Provides style transfer capabilities

* **VAE Processing**: `CompVis/stable-diffusion-v1-4`
  * Manages latent space encoding and decoding
  * Enables visualization of compressed image representations

* **Stylized LoRA Weights**: `training/models-update/(style)_lora/...`
  * Low-rank adaptation (LoRA) is a way to add small changes to an existing model
  * We trained 4 separate LoRAs on 4 distinct styles for optimal image generation
  * Information for each LoRA can be found in the JSON files in their respective folders 

### Training

* In-depth READMEs for training and datasets have been made for your convenience. Please find them below:
  * TRAINING : training/README.md
  * DATASETS : data/README.md

## Project Structure

```
frankAInstein/
├── assets/                 # folder containing image assets application
├── data/
│   ├── datasets/           # folder containing the 3 datasets used for training
│   ├── generate_training_pairs.py    # output data generation for training
│   ├── preprocess.py                 # input data preprocessing
│   └── README.md                     # more information on datasets (sources, generations, etc)
├── src/
│   ├── ai_art_studio.py    # (ARCHIVED - All functions moved to app.py)
│   ├── generate.py         # image and pipeline processing functions
│   └── model.py            # base model/vae loading and management
├── training/
│   ├── models/                    # initial LoRA-based models
│          ├── 2d_animation_lora/
│          │        ├──adapter_config.json        # adapter info 
│          │        ├──adapter_model.safetensor   # stored model
│          │        └──training_info.json         #training info
│          │
│          ├── 3d_animation_lora/     # follows same folder structure as above
│          ├── ghibli_lora/           # ""
│          └── lego_lora/             # ""
│   ├── models-update/             # same structure as models/ but with improved continued learning
│   ├── load_finetuned.py          # loading specific model based on style
│   └── README.md                 # more information on training
│
├── app.py                # main entry point
├── theme2.css              # GUI styling
├── projectNotes.md         # Development notes and story
└── README.md              # This documentation
```

## Dependencies

* **torch**: PyTorch framework for deep learning
* **torchvision**: Computer vision utilities
* **diffusers**: Hugging Face diffusion models library
* **accelerate**: Model acceleration and optimization
* **transformers**: Transformer model support
* **Pillow**: Image processing and manipulation
* **numpy**: Numerical computing
* **scikit-learn**: Machine learning utilities
* **matplotlib**: Data visualization
* **gradio**: Web interface framework


### Learning Objectives

Users will understand:
* How diffusion models process visual information
* The concept of latent space
* The role of noise and denoising in diffusion models

## Safety Features

* **Content Filtering**: Automatic detection of inappropriate content, default images returned instead
* **Predefined Styles**: Limited style options to prevent misuse, no direct prompt access

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
* Reduce image resolution
* Use CPU mode for smaller images
* Close other GPU-intensive applications

**Model Loading Errors**
* Ensure stable internet connection for model downloads
* Check available disk space (models require several GB)
* Verify Python version compatibility

**Import Errors**
* Run `pip install -r requirements.txt` to ensure all dependencies are installed
* Check Python path configuration
* Verify virtual environment activation


## Use of Generative AI Notice
* Deepseek was used for coding assitance and generation

## License

This project is designed for educational purposes. Please ensure compliance with model licenses and usage terms when deploying in educational environments.


This project represents a commitment to making AI education accessible, engaging, and safe for learners of all ages.