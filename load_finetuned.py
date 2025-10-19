"""
Fine-tuned LoRA Model Loading and Management

This module handles the sophisticated loading and management of LoRA (Low-Rank Adaptation) models.
It provides functions to detect, load, and merge fine-tuned models with the base Stable Diffusion pipeline.
"""

import os
import torch
import json
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from peft import PeftModel
import traceback

# Global cache for loaded fine-tuned models to avoid reloading
finetuned_models_cache = {}

def get_available_finetuned_styles(models_dir="training/models"):
    """
    Scan the models directory and identify available fine-tuned LoRA styles.
    
    This function looks for directories containing the required LoRA files:
    - adapter_config.json: Configuration file for the LoRA adapter
    - adapter_model.safetensors: The actual LoRA weights
    
    Args:
        models_dir (str): Path to directory containing LoRA models
        
    Returns:
        list: Sorted list of available style names (e.g., ['ghibli', 'lego', '2d_animation'])
    """
    styles = []
    if not os.path.exists(models_dir):
        print(f"Warning: Fine-tuned models directory not found at {models_dir}")
        return styles

    # Look for directories containing LoRA adapter files
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, "adapter_config.json")
            model_path = os.path.join(item_path, "adapter_model.safetensors")
            
            # Check if this directory contains a valid LoRA model
            if os.path.exists(config_path) and os.path.exists(model_path):
                # Extract style name (e.g., 'ghibli_lora' -> 'ghibli')
                style_name = item.replace("_lora", "")
                styles.append(style_name)
    
    return sorted(styles)

def load_finetuned_model(base_pipe, style_name, models_dir="training/models"):
    """
    Load a specific fine-tuned LoRA model and merge it with the base pipeline.
    Uses caching to avoid reloading the same model multiple times.
    
    Args:
        base_pipe: Base Stable Diffusion pipeline
        style_name (str): Name of the style to load (e.g., 'ghibli')
        models_dir (str): Path to models directory
        
    Returns:
        Pipeline with merged LoRA weights, or base pipeline if loading fails
    """
    global finetuned_models_cache

    full_style_name = f"{style_name}_lora"
    
    # Check cache first to avoid reloading
    if full_style_name in finetuned_models_cache:
        print(f"Returning cached fine-tuned pipeline for {style_name}")
        return finetuned_models_cache[full_style_name]

    lora_path = os.path.join(models_dir, full_style_name)
    if not os.path.exists(lora_path):
        print(f"Error: LoRA model path not found for {style_name} at {lora_path}")
        return base_pipe  # Fallback to base pipeline

    print(f"Loading LoRA for {style_name} from {lora_path}...")
    
    try:
        # Create a copy of the base pipeline to avoid modifying the original
        # This is crucial for maintaining the base model for non-fine-tuned generations
        finetuned_pipe = StableDiffusionImg2ImgPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,  # Will be replaced by merged UNet
            scheduler=base_pipe.scheduler,
            safety_checker=base_pipe.safety_checker,
            feature_extractor=base_pipe.feature_extractor
        )
        
        # Load LoRA weights into the UNet using PEFT library
        lora_model = PeftModel.from_pretrained(finetuned_pipe.unet, lora_path)
        
        # Merge LoRA weights into the base UNet and unload the PeftModel wrapper
        # This creates a single UNet with the LoRA weights permanently integrated
        merged_unet = lora_model.merge_and_unload()
        finetuned_pipe.unet = merged_unet
        
        # Ensure the merged UNet is on the correct device and dtype
        finetuned_pipe.to(base_pipe.device)
        finetuned_pipe.unet.to(dtype=base_pipe.unet.dtype)

        # Cache the loaded pipeline for future use
        finetuned_models_cache[full_style_name] = finetuned_pipe
        print(f"Successfully loaded and merged LoRA for {style_name}")
        return finetuned_pipe
        
    except Exception as e:
        print(f"Failed to load or merge LoRA for {style_name}: {e}")
        traceback.print_exc()
        return base_pipe  # Fallback to base pipeline

def get_training_info(style_name, models_dir="training/models"):
    """
    Load training metadata for a specific fine-tuned model.
    
    Args:
        style_name (str): Name of the style (e.g., 'ghibli')
        models_dir (str): Path to models directory
        
    Returns:
        dict: Training information or None if not found
    """
    full_style_name = f"{style_name}_lora"
    info_path = os.path.join(models_dir, full_style_name, "training_info.json")
    
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading training info for {style_name}: {e}")
    
    return None

def compare_models(base_pipe, finetuned_pipe, image, prompt, device):
    """
    Compare base model vs fine-tuned model outputs side by side.
    Useful for evaluating the effectiveness of fine-tuning.
    
    Args:
        base_pipe: Base Stable Diffusion pipeline
        finetuned_pipe: Fine-tuned pipeline with LoRA weights
        image: PIL Image - Input image
        prompt: str - Text prompt
        device: torch.device - Target device
        
    Returns:
        tuple: (base_result, finetuned_result) PIL Images
    """
    print("Generating with base model...")
    base_result = base_pipe(
        prompt=prompt,
        image=image,
        strength=0.5,
        guidance_scale=5,
        num_inference_steps=25
    ).images[0]
    
    print("Generating with fine-tuned model...")
    finetuned_result = finetuned_pipe(
        prompt=prompt,
        image=image,
        strength=0.5,
        guidance_scale=5,
        num_inference_steps=25
    ).images[0]
    
    return base_result, finetuned_result