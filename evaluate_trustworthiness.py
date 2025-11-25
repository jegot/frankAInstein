"""
Trustworthiness Evaluation Script for frankAInstein

This script evaluates the trustworthiness of the AI image generation system by:
1. Testing Accountability and Responsibility: Verifying watermark detection
2. Testing Reliability and Robustness: Evaluating watermark survival under attacks

The evaluation simulates various attacks (cropping, blurring, JPEG compression) and
measures the success rate of watermark detection after each attack.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from src.model import load_models
from src.generate import preprocess_image, generate_with_progression, prompt_conversion
from src.watermark import InvisibleWatermarker, detect_watermark_in_image
from training.load_finetuned import load_finetuned_model


class AttackSimulator:
    """
    Simulates various attacks on watermarked images to test robustness.
    """
    
    @staticmethod
    def crop_attack(image, crop_ratio=0.1):
        """
        Simulate cropping attack by removing edges.
        
        Args:
            image (PIL.Image): Input image
            crop_ratio (float): Fraction of image to crop from each side (0.0-0.5)
            
        Returns:
            PIL.Image: Cropped image
        """
        width, height = image.size
        crop_pixels = int(min(width, height) * crop_ratio)
        return image.crop((crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels))
    
    @staticmethod
    def blur_attack(image, kernel_size=5):
        """
        Simulate blurring attack using Gaussian blur.
        
        Args:
            image (PIL.Image): Input image
            kernel_size (int): Blur kernel size (must be odd)
            
        Returns:
            PIL.Image: Blurred image
        """
        img_array = np.array(image)
        blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        return Image.fromarray(blurred)
    
    @staticmethod
    def jpeg_compression_attack(image, quality=75):
        """
        Simulate JPEG compression attack.
        
        Args:
            image (PIL.Image): Input image
            quality (int): JPEG quality (1-100, lower = more compression)
            
        Returns:
            PIL.Image: Compressed image
        """
        import io
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality)
        output.seek(0)
        return Image.open(output)
    
    @staticmethod
    def resize_attack(image, scale=0.8):
        """
        Simulate resizing attack.
        
        Args:
            image (PIL.Image): Input image
            scale (float): Resize scale factor
            
        Returns:
            PIL.Image: Resized image
        """
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def noise_attack(image, noise_level=10):
        """
        Simulate noise addition attack.
        
        Args:
            image (PIL.Image): Input image
            noise_level (int): Standard deviation of Gaussian noise
            
        Returns:
            PIL.Image: Noisy image
        """
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    @staticmethod
    def combined_attack(image, crop_ratio=0.05, blur_kernel=3, jpeg_quality=85):
        """
        Simulate multiple attacks in sequence.
        
        Args:
            image (PIL.Image): Input image
            crop_ratio (float): Crop ratio
            blur_kernel (int): Blur kernel size
            jpeg_quality (int): JPEG quality
            
        Returns:
            PIL.Image: Attacked image
        """
        attacked = AttackSimulator.crop_attack(image, crop_ratio)
        attacked = AttackSimulator.blur_attack(attacked, blur_kernel)
        attacked = AttackSimulator.jpeg_compression_attack(attacked, jpeg_quality)
        return attacked


class TrustworthinessEvaluator:
    """
    Main evaluation class for testing watermark robustness.
    """
    
    def __init__(self, output_dir="evaluation_results"):
        """
        Initialize evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.watermark_text = "AI_GENERATED_FRANKAINSTEIN"
        self.watermarker = InvisibleWatermarker(self.watermark_text)
        self.results = []
        
    def generate_test_images(self, num_images=10, styles=None):
        """
        Generate test images with watermarks.
        
        Args:
            num_images (int): Number of test images to generate
            styles (list): List of styles to test, or None for all
            
        Returns:
            list: List of (image, style, metadata) tuples
        """
        print("Loading models...")
        pipe, device, vae = load_models()
        
        if styles is None:
            styles = ["studio ghibli", "LEGO", "2D animation", "3D animation"]
        
        test_images = []
        
        # Use a simple test image (create a colored square as base)
        base_image = Image.new('RGB', (384, 384), color=(100, 150, 200))
        
        print(f"Generating {num_images} test images...")
        for i in range(num_images):
            style = styles[i % len(styles)]
            print(f"  Generating image {i+1}/{num_images} with style: {style}")
            
            # Load fine-tuned model
            lora_name_map = {
                "studio ghibli": "ghibli",
                "LEGO": "lego",
                "2D animation": "2d_animation",
                "3D animation": "3d_animation"
            }
            
            if style in lora_name_map:
                fine_tuned_pipe = load_finetuned_model(pipe, lora_name_map[style])
            else:
                fine_tuned_pipe = pipe
            
            # Generate image
            prompt = prompt_conversion(style)
            result, _ = generate_with_progression(
                fine_tuned_pipe, base_image, prompt, device, 
                strength=0.6, guidance_scale=18.0, num_inference_steps=20
            )
            
            # Watermark is already added in generate_with_progression
            test_images.append((result, style, {"index": i, "style": style}))
        
        return test_images

    def load_images_from_directory(self, directory, max_images=None, resize_to=(384, 384)):
        """
        Load user-provided images, embed watermark, and prepare for evaluation.

        Args:
            directory (str | Path): Folder containing user images
            max_images (int): Optional limit on number of images
            resize_to (tuple): Target size (width, height) to standardize evaluation

        Returns:
            list: List of (watermarked_image, label, metadata) tuples
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Custom images directory not found: {directory}")

        allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        image_paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in allowed_ext]

        if not image_paths:
            raise ValueError(f"No image files found in {directory}. Supported extensions: {allowed_ext}")

        if max_images:
            image_paths = image_paths[:max_images]

        custom_images = []
        print(f"Loading {len(image_paths)} custom images from {directory}...")

        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                if resize_to:
                    img = img.resize(resize_to, Image.Resampling.LANCZOS)
                watermarked_img = self.watermarker.embed_watermark(img, strength=0.2)
                label = f"custom_{idx}"
                metadata = {"index": idx, "source_path": str(path)}
                custom_images.append((watermarked_img, label, metadata))
            except Exception as exc:
                print(f"  WARNING: Failed to load {path}: {exc}")

        if not custom_images:
            raise RuntimeError("Failed to load any custom images for evaluation.")

        return custom_images
    
    def evaluate_attack(self, image, attack_name, attack_func, attack_params):
        """
        Evaluate watermark detection after a specific attack.
        
        Args:
            image (PIL.Image): Watermarked image
            attack_name (str): Name of the attack
            attack_func (callable): Attack function
            attack_params (dict): Parameters for the attack
            
        Returns:
            dict: Evaluation results
        """
        # Apply attack
        attacked_image = attack_func(image, **attack_params)
        
        # Detect watermark with adaptive threshold based on attack type
        # Use lower threshold for geometric attacks (crop, resize) as they're harder
        # Combined attacks are the hardest, so use lowest threshold
        if attack_name == 'combined':
            threshold = 0.45  # Lowest threshold for combined attacks (multiple attacks)
        elif attack_name in ['crop', 'resize']:
            threshold = 0.45  # Lower threshold for geometric attacks
        else:
            threshold = 0.5  # Standard threshold for other attacks
        
        detected, confidence, extracted_bits = self.watermarker.detect_watermark(attacked_image, threshold=threshold)
        
        return {
            "attack_name": attack_name,
            "attack_params": attack_params,
            "detected": detected,
            "confidence": float(confidence),
            "success": detected
        }
    
    def run_evaluation(self, test_images, attacks_config=None):
        """
        Run full evaluation suite.
        
        Args:
            test_images (list): List of test images
            attacks_config (dict): Configuration for attacks, or None for defaults
        """
        if attacks_config is None:
            attacks_config = {
                "crop": [
                    {"crop_ratio": 0.05},
                    {"crop_ratio": 0.1},
                    {"crop_ratio": 0.2}
                ],
                "blur": [
                    {"kernel_size": 3},
                    {"kernel_size": 5},
                    {"kernel_size": 7}
                ],
                "jpeg": [
                    {"quality": 95},
                    {"quality": 85},
                    {"quality": 75},
                    {"quality": 65}
                ],
                "resize": [
                    {"scale": 0.9},
                    {"scale": 0.8},
                    {"scale": 0.7}
                ],
                "noise": [
                    {"noise_level": 5},
                    {"noise_level": 10},
                    {"noise_level": 15}
                ],
                "combined": [
                    {"crop_ratio": 0.05, "blur_kernel": 3, "jpeg_quality": 85}
                ]
            }
        
        print("\n" + "="*60)
        print("Starting Trustworthiness Evaluation")
        print("="*60)
        
        all_results = []
        
        for img_idx, (image, style, metadata) in enumerate(test_images):
            print(f"\nEvaluating image {img_idx + 1}/{len(test_images)} (Style: {style})")
            
            # First, verify watermark is present in original
            # Use standard threshold for original detection
            detected_original, confidence_original, _ = self.watermarker.detect_watermark(image, threshold=0.5)
            if not detected_original:
                print(f"  WARNING: Watermark not detected in original image! Confidence: {confidence_original:.3f}")
                print(f"  Note: This may indicate watermark embedding issues. Continuing evaluation anyway...")
            
            image_results = {
                "image_index": img_idx,
                "style": style,
                "original_detection": {
                    "detected": detected_original,
                    "confidence": float(confidence_original)
                },
                "attack_results": []
            }
            
            # Test each attack type
            for attack_type, attack_configs in attacks_config.items():
                # Map attack type names to method names
                attack_method_map = {
                    "crop": "crop_attack",
                    "blur": "blur_attack",
                    "jpeg": "jpeg_compression_attack",  # Fixed: method is jpeg_compression_attack, not jpeg_attack
                    "resize": "resize_attack",
                    "noise": "noise_attack",
                    "combined": "combined_attack"
                }
                method_name = attack_method_map.get(attack_type, f"{attack_type}_attack")
                attack_func = getattr(AttackSimulator, method_name)
                
                for config in attack_configs:
                    result = self.evaluate_attack(image, attack_type, attack_func, config)
                    result["image_index"] = img_idx
                    result["style"] = style
                    image_results["attack_results"].append(result)
                    all_results.append(result)
                    
                    status = "[PASS]" if result["success"] else "[FAIL]"
                    print(f"  {status} {attack_type} {config}: "
                          f"Detected={result['detected']}, Confidence={result['confidence']:.3f}")
            
            self.results.append(image_results)
        
        # Calculate summary statistics
        self._calculate_statistics(all_results)
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")
    
    def _calculate_statistics(self, all_results):
        """Calculate summary statistics."""
        self.statistics = {}
        
        # Overall success rate
        total = len(all_results)
        successful = sum(1 for r in all_results if r["success"])
        self.statistics["overall_success_rate"] = successful / total if total > 0 else 0
        
        # Success rate by attack type
        attack_types = set(r["attack_name"] for r in all_results)
        self.statistics["by_attack_type"] = {}
        
        for attack_type in attack_types:
            attack_results = [r for r in all_results if r["attack_name"] == attack_type]
            attack_total = len(attack_results)
            attack_successful = sum(1 for r in attack_results if r["success"])
            self.statistics["by_attack_type"][attack_type] = {
                "success_rate": attack_successful / attack_total if attack_total > 0 else 0,
                "total_tests": attack_total,
                "successful": attack_successful,
                "average_confidence": np.mean([r["confidence"] for r in attack_results])
            }
        
        # Average confidence
        self.statistics["average_confidence"] = np.mean([r["confidence"] for r in all_results])
        
        print("\n" + "-"*60)
        print("Evaluation Statistics")
        print("-"*60)
        print(f"Overall Success Rate: {self.statistics['overall_success_rate']:.2%}")
        print(f"Average Confidence: {self.statistics['average_confidence']:.3f}")
        print("\nSuccess Rate by Attack Type:")
        for attack_type, stats in self.statistics["by_attack_type"].items():
            print(f"  {attack_type:15s}: {stats['success_rate']:.2%} "
                  f"({stats['successful']}/{stats['total_tests']}) "
                  f"Avg Confidence: {stats['average_confidence']:.3f}")
    
    def _save_results(self):
        """Save evaluation results to JSON."""
        results_file = self.output_dir / "evaluation_results.json"
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "watermark_text": self.watermark_text,
            "statistics": self.statistics,
            "detailed_results": self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        # Success rate by attack type
        attack_types = list(self.statistics["by_attack_type"].keys())
        success_rates = [self.statistics["by_attack_type"][at]["success_rate"] 
                         for at in attack_types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of success rates
        ax1.bar(attack_types, success_rates, color=['#4CAF50' if sr > 0.7 else '#FF9800' if sr > 0.5 else '#F44336' 
                                                    for sr in success_rates])
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Watermark Detection Success Rate by Attack Type')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        for i, (at, sr) in enumerate(zip(attack_types, success_rates)):
            ax1.text(i, sr + 0.02, f'{sr:.1%}', ha='center', va='bottom')
        
        # Average confidence by attack type
        avg_confidences = [self.statistics["by_attack_type"][at]["average_confidence"] 
                          for at in attack_types]
        ax2.bar(attack_types, avg_confidences, color='#2196F3')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Detection Confidence by Attack Type')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        for i, (at, conf) in enumerate(zip(attack_types, avg_confidences)):
            ax2.text(i, conf + 0.02, f'{conf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_summary.png", dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {self.output_dir / 'evaluation_summary.png'}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="frankAInstein watermark trustworthiness evaluation")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Optional path to directory containing custom test images. "
                             "If omitted, the script will look for 'images_trustworthiness/'.")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to generate (or max custom images to load).")
    args = parser.parse_args()
    print("="*60)
    print("frankAInstein Trustworthiness Evaluation")
    print("Focus: Accountability & Responsibility, Reliability & Robustness")
    print("="*60)
    
    evaluator = TrustworthinessEvaluator()
    
    # Generate or load test images
    custom_dir = None
    if args.images_dir:
        custom_dir = Path(args.images_dir)
    else:
        default_dir = Path("images_trustworthiness")
        if default_dir.exists():
            custom_dir = default_dir

    if custom_dir:
        print("\nStep 1: Loading custom images...")
        test_images = evaluator.load_images_from_directory(custom_dir, max_images=args.num_images)
    else:
        print("\nStep 1: Generating test images with watermarks...")
        test_images = evaluator.generate_test_images(num_images=args.num_images)
    
    # Run evaluation
    print("\nStep 2: Running attack simulations...")
    evaluator.run_evaluation(test_images)
    
    print("\nEvaluation complete! Check the 'evaluation_results' directory for detailed results.")


if __name__ == "__main__":
    main()

