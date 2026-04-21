"""
Preprocess scratch-assay images for ImageJ Wound_healing_size_tool.

For each image:
  1. Detect the circular FOV.
  2. Flat-field correction: subtract large Gaussian blur (sigma=61) to remove vignetting.
  3. Replace ALL dark pixels (inside and outside FOV) with noise so ImageJ
     does not mistake dark vignetting areas for wound.
  4. Save to  Vik_preprocessed/
"""

import os, sys
import numpy as np
import cv2
from skimage.io import imread, imsave

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(PROJECT_DIR, "Vik")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "Vik_preprocessed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_fov_mask(gray):
    h, w = gray.shape
    _, rough = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * 0.95), 255, -1)
    return mask


def process(filepath, output_dir):
    name = os.path.basename(filepath)
    gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Step 1: flat-field correction — removes vignetting gradient
    bg   = cv2.GaussianBlur(gray, (0, 0), 61)
    flat = gray - bg

    # Step 2: detect FOV on flat-corrected image
    fov_mask = find_fov_mask(np.clip(flat + 128, 0, 255).astype(np.uint8))

    # Step 3: replace ALL dark pixels with noise sampled from bright cell region
    fov_vals    = flat[fov_mask == 255]
    dark_thresh = np.percentile(fov_vals, 8)
    bright_vals = fov_vals[fov_vals > dark_thresh]
    noise_mean  = float(bright_vals.mean())
    noise_std   = max(float(bright_vals.std()), 3.0)

    rng   = np.random.default_rng(seed=42)
    noise = rng.normal(loc=noise_mean, scale=noise_std, size=flat.shape).astype(np.float32)

    dark_px    = flat < dark_thresh
    flat[dark_px] = noise[dark_px]

    # Step 4: normalize to 0-255 uint8
    lo, hi = np.percentile(flat[fov_mask == 255], [1, 99])
    flat   = np.clip((flat - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)

    out_path = os.path.join(output_dir, name)
    imsave(out_path, flat)
    return name


def main():
    files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )
    if not files:
        print(f"No images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Preprocessing {len(files)} images -> {OUTPUT_DIR}\n")
    for i, fp in enumerate(files, 1):
        name = process(fp, OUTPUT_DIR)
        print(f"  [{i:3d}/{len(files)}] {name}")

    print(f"\nDone. Open images in  Vik_preprocessed/  with ImageJ.")


if __name__ == "__main__":
    main()
