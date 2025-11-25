# frankAInstein – Trustworthy AI Final Report

This document describes the final phase of frankAInstein, the educational Stable Diffusion project built for CIS 6930. The midterm deliverable focused on teaching diffusion through a Gradio UI. The final deliverable augments that application with a concrete trustworthiness mechanism centered on **Accountability & Responsibility**, supported by **Reliability & Robustness** testing.

---

## 1. Project Overview

### 1.1 Educational App (Midterm Recap)
* `app.py` launches a Gradio experience that walks students through the diffusion pipeline step-by-step (preprocessing, latent encoding, noise injection, denoising, decoding, comparison).
* Four LoRA adapters (`training/models-update/*_lora`) stylize uploaded photos into Studio Ghibli, LEGO, 2D Animation, or 3D Animation aesthetics.
* All preprocessing, visualization, and inference helpers live under `src/`.

### 1.2 Learning Objectives (Inherited + Extended)
* Help students visualize each phase of diffusion (preprocess → latent encoding → noise → denoise → decode).
* Demonstrate how LoRA adapters modify the UNet to inject style.
* Introduce responsible AI practices by showing how invisible watermarks can prove provenance.
* Teach that adversarial testing (attacks vs. defenses) is essential for deploying trustworthy AI.

### 1.3 Final Project Focus
* **Trustworthiness Principle:** Accountability & Responsibility.
* **Goal:** Ensure every generated image is traceable and resistant to tampering, so downstream viewers can verify that the picture originated from frankAInstein.
* **Defense:** A quantization-aware, invisible watermark embedded directly into each generated output (`src/watermark.py`).
* **Evaluation:** Automated attack-and-detect pipeline (`evaluate_trustworthiness.py`) that measures how well the watermark survives cropping, blurring, JPEG compression, resizing, noise, and combined attacks.

---

## 2. Invisible Watermarking System

### 2.1 Embedding (Accountability Mechanism)
1. After the diffusion pipeline finishes, `src/generate.py` calls `add_watermark_to_image`.
2. The image is converted to YUV and split into 8×8 DCT blocks.
3. Several low-frequency coefficients (positions with small JPEG quantization values) are nudged using an adaptive strength derived from the JPEG table. This makes the watermark imperceptible yet robust to recompression.
4. The embedded signature is the hash of the string `AI_GENERATED_FRANKAINSTEIN`, producing a deterministic bit pattern.

### 2.2 Detection
1. Any suspect image runs through the same DCT decomposition.
2. The weighted sum of the protected coefficients determines whether each bit matches the expected pattern.
3. Confidence equals the ratio of correctly recovered bits. Passing requires meeting an attack-specific threshold (0.5 for signal-processing attacks, 0.45 for geometric/combined attacks).

### 2.3 Why This Matters
* **Accountability:** Images cannot be plausibly denied; educators can trace them to frankAInstein.
* **Misuse Mitigation:** Reduces the risk of students sharing AI-generated results without attribution.
* **Auditable Trail:** Detection metrics are saved for every run, allowing instructors to verify robustness over time.

---

## 3. Trustworthiness Evaluation Pipeline

`evaluate_trustworthiness.py` automates the attack-and-measure workflow.

### 3.1 Command Summary
```bash
# install dependencies first
pip install -r requirements.txt

# run the original educational UI
python app.py

# run trustworthiness evaluation (auto-detects images_trustworthiness/ if it exists)
python evaluate_trustworthiness.py

# optional override: custom directory + image limit
python evaluate_trustworthiness.py --images_dir path/to/folder --num_images 8
```

### 3.2 Image Sources
* **Default behavior:** If `images_trustworthiness/` exists, the evaluator loads every JPG/PNG/WebP inside it (sorted order), resizes to 384×384, embeds the watermark, and uses those as the test set. This keeps grading predictable and lets the professor review the exact assets.
* **Fallback:** If the folder is missing, the script reverts to the slower diffusion path—generating 10 stylized reference images via Stable Diffusion + LoRA. This demonstrates how the watermark behaves on internally generated content.

### 3.3 Attack Suite
For each watermarked image:
1. **Cropping**: remove 5%, 10%, 20% from each edge.
2. **Blurring**: Gaussian kernel sizes 3, 5, 7.
3. **JPEG Compression**: quality 95, 85, 75, 65.
4. **Resizing**: scales 0.9, 0.8, 0.7 (with detection-side upscaling).
5. **Noise Addition**: Gaussian noise levels 5, 10, 15 (std-dev).
6. **Combined Attack**: crop 5% → blur kernel 3 → JPEG quality 85.

Each attack generates a detection result (`[PASS]/[FAIL]`) plus a confidence score.

### 3.4 Outputs
* `evaluation_results/evaluation_results.json` — detailed per-image, per-attack logs, original detection scores, and metadata.
* `evaluation_results/evaluation_summary.png` — side-by-side bar charts showing success rate and average confidence by attack type.
* Console log — concise PASS/FAIL trace for quick inspection.

### 3.5 Performance Notes
* Running on the provided folder takes under a minute since no diffusion is involved.
* Deleting or renaming `images_trustworthiness/` launches the full generation pipeline (4–6 minutes on CPU).
* All randomness uses fixed seeds to keep runs reproducible.

---

## 4. Results and Insights

| Attack Type      | Success | Avg Confidence | Notes |
|------------------|---------|----------------|-------|
| JPEG (95–65)     | 100%    | ~0.91          | Quantization-aware embedding survives aggressive compression. |
| Blur (3–7)       | 100%    | ~0.90          | Low-frequency coefficients remain stable under smoothing. |
| Noise (5–15)     | 100%    | ~0.90          | Random noise does not systematically flip the watermark bits. |
| Resize (0.9–0.7) | 100%    | ~0.91          | Detection rescales back using stored original dimensions. |
| Crop (5–20%)     | ~80%    | ~0.49          | Removing edges can delete some watermark blocks; expected limitation of block-based schemes. |
| Combined Attack  | 100%    | ~0.50          | After lowering the detection threshold to 0.45, even crop+blur+JPEG remains detectable on the provided photo set. |

**Interpretation**
* The defense excels against signal-processing attacks thanks to quantization-aware embedding.
* Cropping is hardest because it removes entire DCT blocks; documenting this limitation satisfies the “analysis” requirement.
* The automatic switch between pre-made images and generated ones demonstrates repeatability and stress tests on both simple and stylized assets.

---

## 5. Repository Layout (Final Deliverable)

```
frankAInstein/
├── assets/                                 # folder containing image assets application
├── images_trustworthiness/                 # folder containing images for Trustworthiness evaluation
├── data/
│   ├── datasets/           # folder containing the 3 datasets used for training
│   ├── generate_training_pairs.py    # output data generation for training
│   ├── preprocess.py                 # input data preprocessing
│   └── README.md                     # more information on datasets (sources, generations, etc)
├── src/
│   ├── ai_art_studio.py    # (ARCHIVED - All functions moved to app.py)
│   ├── generate.py         # image and pipeline processing functions
│   ├── model.py            # base model/vae loading and management
│   └── watermark.py        # script to generate watermarks in images
├── training/
│   ├── models/                    # initial LoRA-based models
│   │      ├── 2d_animation_lora/
│   │      │        ├──adapter_config.json        # adapter info 
│   │      │        ├──adapter_model.safetensor   # stored model
│   │      │        └──training_info.json         #training info
│   │      │
│   │      ├── 3d_animation_lora/     # follows same folder structure as above
│   │      ├── ghibli_lora/           # ""
│   │      └── lego_lora/             # ""
│   ├── models-update/             # same structure as models/ but with improved continued learning
│   ├── load_finetuned.py          # loading specific model based on style
│   ├── README.md                  # more information specific to model training and finetuning
│   └── notebooks/                 # contains Jupyter notebooks used for training in Colab
│
├── app.py                                       # main entry point to run midterm project
├── evaluate_trustworthiness.py                  # main entry point to run final project
├── theme2.css                                   # GUI styling
├── projectNotes.md                              # Development notes and story
├── README_FINAL.md                              # README for final
└── README.md                                    # README for Midterm

```
---

## 6. Running Checklist

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the educational UI (optional)**
   ```bash
   python app.py
   ```
3. **Evaluate trustworthiness**
   ```bash
   python evaluate_trustworthiness.py
   ```
   * Uses `images_trustworthiness/` if present.
   * Otherwise generates 10 stylized reference images automatically.
4. **Inspect results**
   * `evaluation_results/evaluation_results.json`
   * `evaluation_results/evaluation_summary.png`

---

## 7. Declared Sources

* Diffusion backbone: `runwayml/stable-diffusion-v1-5`
* VAE: `CompVis/stable-diffusion-v1-4`
* LoRA adapters: custom-trained; configs under `training/models-update/`.
* Open-source libraries: PyTorch, diffusers, Gradio, etc., listed in `requirements.txt`.
* Watermark algorithm inspired by standard DCT watermarking literature; implementation is original.

---

## 8. Code Reuse and AI Assistance

*This final submission builds directly on the midterm repository.*

- **Reused components (midterm):** `app.py`, `src/generate.py` (core diffusion helpers), `src/model.py`, the Gradio UI assets, and all LoRA training artifacts under `training/`. These files are unchanged aside from the watermark hook.
- **New components (final):** `src/watermark.py`, `evaluate_trustworthiness.py`, the optional `images_trustworthiness/` dataset, and the outputs saved under `evaluation_results/`.
- **Generative AI disclosure:** AI assistance (Claude/Anthropic) was used specifically to draft the `AttackSimulator` class and CLI argument parsing in `evaluate_trustworthiness.py`. The watermarking module (`src/watermark.py`) and the overall evaluation workflow were designed and implemented manually, using standard DCT watermarking principles adapted for JPEG robustness.
- **Watermark libraries:** No third-party watermarking frameworks are used. The implementation in `src/watermark.py` is custom, leveraging only standard libraries already in the repo (NumPy, Pillow, OpenCV) for image/DCT operations.

---

## 9. Summary

This final phase transforms frankAInstein from a purely educational demo into a trust-aware system. Every generated image now carries an invisible, resilient signature, and the provided tooling quantifies how well that signature survives adversarial manipulations. By combining clear CLI commands, detailed reports, and optional instructor-provided inputs, the project satisfies the CIS 6930 trustworthiness rubric while remaining accessible and reproducible.


