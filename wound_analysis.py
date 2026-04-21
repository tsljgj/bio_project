"""
Wound area analysis — algorithm ported from Archive/analysis.py.

Key steps:
  1. FOV mask via largest contour + minEnclosingCircle.
  2. Flat-field correction: subtract Gaussian blur (sigma=61) → removes vignetting.
  3. Texture map: local_std + 0.35 * Sobel gradient magnitude.
  4. Scratch detection in the inner 78% of FOV; scored by area + height + centrality.
  5. Expansion step to recover missed wound edges.
  6. Annotate original image (blue overlay) and save; write CSV.
"""

import os, sys, csv, datetime
import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from skimage.io import imread

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR   = os.path.join(PROJECT_DIR, "Vik")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "results")
LABELED_DIR = os.path.join(OUTPUT_DIR, "labeled")
CSV_PATH    = os.path.join(OUTPUT_DIR, "wound_areas.csv")
UM_PER_PX   = 840 / 557   # from archive's calibration

os.makedirs(LABELED_DIR, exist_ok=True)


# ── FOV detection ─────────────────────────────────────────────────────────────

def get_fov_mask(img):
    h, w = img.shape
    _, rough = cv2.threshold(img.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask, (w // 2, h // 2), min(h, w) // 2
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * 0.95), 255, -1)
    return mask, (int(cx), int(cy)), int(r)


# ── Flat-field correction + dark-pixel removal ───────────────────────────────

def preprocess_image(img, fov_mask):
    img = img.astype(np.float32)
    bg   = cv2.GaussianBlur(img, (0, 0), 61)   # slow gradient = background
    flat = img - bg                              # subtract → flat illumination

    # Replace ALL dark pixels (inside and outside FOV) with noise so they
    # don't masquerade as wound.  Threshold = 8th percentile of FOV pixels.
    fov_vals = flat[fov_mask == 255]
    if len(fov_vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    dark_thresh = np.percentile(fov_vals, 8)
    bright_vals = fov_vals[fov_vals > dark_thresh]
    noise_mean  = float(bright_vals.mean())
    noise_std   = max(float(bright_vals.std()), 1.0)
    rng   = np.random.default_rng(seed=42)
    noise = rng.normal(loc=noise_mean, scale=noise_std, size=flat.shape).astype(np.float32)
    dark_px = flat < dark_thresh
    flat[dark_px] = noise[dark_px]

    lo, hi = np.percentile(flat[fov_mask == 255], [1, 99])
    flat = np.clip((flat - lo) / (hi - lo + 1e-6), 0, 1)
    return flat.astype(np.float32)


# ── Texture map ───────────────────────────────────────────────────────────────

def compute_texture_map(img, mask, block_size=31):
    mask_f    = (mask == 255).astype(np.float32)
    img_m     = img * mask_f
    local_sum = uniform_filter(img_m, size=block_size)
    local_cnt = np.maximum(uniform_filter(mask_f, size=block_size), 1e-6)
    local_mean = local_sum / local_cnt
    local_sq  = uniform_filter(img_m ** 2, size=block_size)
    local_var = np.maximum(local_sq / local_cnt - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)
    gx  = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx*gx + gy*gy), (0, 0), 3)
    tex = local_std + 0.35 * grad
    tex[mask == 0] = 0
    vals = tex[mask == 255]
    if len(vals) == 0:
        return np.zeros_like(tex, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    tex = np.clip((tex - lo) / (hi - lo + 1e-6), 0, 1)
    tex[mask == 0] = 0
    return tex.astype(np.float32)


# ── Multi-representation wound probability map ────────────────────────────────

def build_representations(img, mask):
    """
    Build five complementary wound-probability maps, each in [0,1] where
    1.0 = very likely wound (cell-free gap).

    Returns a dict of named float32 maps and the flat-corrected image.
    """
    flat = preprocess_image(img, mask)

    # Rep 1: inverted intensity — wound gaps are darker after flat correction
    rep_intensity = 1.0 - flat

    # Rep 2: inverted texture (local_std + Sobel) — wound has low texture
    texture = compute_texture_map(flat, mask, block_size=31)
    rep_texture = 1.0 - texture

    # Rep 3: inverted fine texture at smaller block — catches narrow gaps
    texture_fine = compute_texture_map(flat, mask, block_size=15)
    rep_texture_fine = 1.0 - texture_fine

    # Rep 4: inverted Laplacian energy — cells have sharp membranes, wound does not
    lap = cv2.Laplacian(flat, cv2.CV_32F, ksize=5)
    lap_energy = cv2.GaussianBlur(np.abs(lap), (0, 0), 5)
    vals = lap_energy[mask == 255]
    if len(vals):
        lo, hi = np.percentile(vals, [1, 99])
        lap_energy = np.clip((lap_energy - lo) / (hi - lo + 1e-6), 0, 1)
    rep_lap = 1.0 - lap_energy
    rep_lap[mask == 0] = 0

    # Rep 5: local entropy proxy — wound region has low entropy (uniform)
    # Approximated as inverted local range (max - min in a neighborhood)
    from scipy.ndimage import maximum_filter, minimum_filter
    local_max = maximum_filter(flat, size=21)
    local_min = minimum_filter(flat, size=21)
    local_range = local_max - local_min
    vals = local_range[mask == 255]
    if len(vals):
        lo, hi = np.percentile(vals, [1, 99])
        local_range = np.clip((local_range - lo) / (hi - lo + 1e-6), 0, 1)
    rep_range = 1.0 - local_range
    rep_range[mask == 0] = 0

    return {
        'intensity':     rep_intensity,
        'texture':       rep_texture,
        'texture_fine':  rep_texture_fine,
        'laplacian':     rep_lap,
        'local_range':   rep_range,
    }, flat, texture


# ── Scratch detection ─────────────────────────────────────────────────────────

def detect_scratch(img, mask, cx, cy, r):
    reps, flat, texture = build_representations(img, mask)

    # Only search in inner 78% of FOV to avoid edge artifacts
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), int(r * 0.78), 255, -1)
    valid = (mask == 255) & (inner == 255)

    # Weighted ensemble — texture signal is most reliable for scratch assays
    weights = {
        'intensity':    0.15,
        'texture':      0.35,
        'texture_fine': 0.25,
        'laplacian':    0.15,
        'local_range':  0.10,
    }
    ensemble = np.zeros_like(flat)
    for name, w in weights.items():
        ensemble += w * reps[name]
    ensemble[mask == 0] = 0

    scratch_score = cv2.GaussianBlur(ensemble, (0, 0), 7)
    vals = scratch_score[valid]
    if len(vals) == 0:
        return np.zeros_like(mask), texture, 0, 'bad_fov', reps

    # Top 25% of ensemble score = wound candidates
    thresh = np.percentile(vals, 75)
    cand = ((scratch_score >= thresh) & valid).astype(np.uint8) * 255
    # Rectangular kernels suit a vertical scratch band
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71)))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    best_label, best_score = None, -1e18
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        c_x, c_y = cents[i]
        if area < 5000:
            continue
        score = area + 2.5 * bh - 1.2 * abs(c_x - cx) - 0.3 * abs(c_y - cy)
        if score > best_score:
            best_score, best_label = score, i

    if best_label is None:
        return np.zeros_like(mask), texture, 0, 'no_component', reps

    scratch_mask = (labels == best_label).astype(np.uint8) * 255

    # Expansion step — recover wound edges using ensemble score instead of
    # single texture, making the boundary decision more robust
    expand_thresh = np.percentile(ensemble[valid], 60)
    expand_cand   = ((ensemble >= expand_thresh) & valid).astype(np.uint8) * 255
    dilated       = cv2.dilate(scratch_mask,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    expanded      = ((dilated > 0) & (expand_cand > 0)).astype(np.uint8) * 255
    scratch_mask  = cv2.bitwise_or(scratch_mask, expanded)
    scratch_mask  = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81)))
    scratch_mask[mask == 0] = 0

    scratch_area_px = int((scratch_mask > 0).sum())
    return scratch_mask, texture, scratch_area_px, 'ok', reps


# ── Width measurement ─────────────────────────────────────────────────────────

def measure_width(scratch_mask):
    rows, widths = [], []
    for y in np.where(np.any(scratch_mask > 0, axis=1))[0]:
        xs = np.where(scratch_mask[y] > 0)[0]
        if len(xs) < 10:
            continue
        rows.append(y)
        widths.append(xs[-1] - xs[0] + 1)
    if len(widths) < 20:
        return np.nan, np.nan
    widths = np.array(widths, dtype=np.float32)
    med  = np.median(widths)
    keep = (widths > 0.4 * med) & (widths < 2.2 * med)
    w    = widths[keep]
    if len(w) < 10:
        return np.nan, np.nan
    return float(np.mean(w) * UM_PER_PX), float(np.std(w) * UM_PER_PX)


# ── Annotate and save ─────────────────────────────────────────────────────────

def annotate_and_save(filepath, scratch_mask, fov_mask, scratch_area_px,
                      width_um, width_std_um, qc_flag):
    name = os.path.basename(filepath)
    raw  = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Dim the region outside FOV
    outside = fov_mask <= 127
    overlay[outside] *= 0.15

    # Blue overlay on scratch
    scratch_px = scratch_mask > 127
    overlay[scratch_px] = (overlay[scratch_px] * 0.35
                           + np.array([40, 40, 220], dtype=np.float32) * 0.65)

    # Cyan contour on scratch edge
    edge = cv2.morphologyEx((scratch_mask > 127).astype(np.uint8) * 255,
                             cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    overlay[edge > 0] = [0, 255, 255]

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Text labels
    now   = datetime.datetime.now()
    w_str = f"{width_um:.0f} +/- {width_std_um:.0f} um" if (width_um and np.isfinite(width_um)) else "n/a"
    lines = [
        f"File: {name}",
        f"Wound Area: {scratch_area_px:,} px",
        f"Scratch Width: {w_str}",
        f"QC: {qc_flag}   Date: {now.day}/{now.month}/{now.year}",
    ]
    h = overlay.shape[0]
    scale = max(h / 1200, 1.0)
    thick = max(int(2 * scale), 2)
    for i, text in enumerate(lines):
        y = int(50 * scale) + i * int(45 * scale)
        cv2.putText(overlay, text, (4, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85*scale, (0,0,0),     thick+1, cv2.LINE_AA)
        cv2.putText(overlay, text, (2, y),   cv2.FONT_HERSHEY_SIMPLEX,
                    0.85*scale, (0,255,255), thick,   cv2.LINE_AA)

    out_path = os.path.join(LABELED_DIR, "lb_" + name)
    cv2.imwrite(out_path, overlay)


# ── Main ──────────────────────────────────────────────────────────────────────

def save_debug_panel(filepath, reps, scratch_mask, fov_mask):
    """Save a 3×2 panel showing all representations + final mask."""
    raw   = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    names = ['intensity', 'texture', 'texture_fine', 'laplacian', 'local_range']
    tiles = [raw]
    for n in names:
        tiles.append((reps[n] * 255).clip(0, 255).astype(np.uint8))
    # Final mask as last tile
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    overlay[scratch_mask > 0] = (0, 80, 220)
    overlay[fov_mask == 0]    = (20, 20, 20)
    tiles.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY))  # keep grayscale for grid

    h, w = raw.shape
    labels = ['Raw', 'Intensity', 'Texture-31', 'Texture-15', 'Laplacian', 'LocalRange', 'Wound mask']
    # Pad to 8 tiles (2 rows × 4 cols)
    while len(tiles) < 8:
        tiles.append(np.zeros((h, w), dtype=np.uint8))
        labels.append('')

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(h / 1200, 0.6)
    rows = []
    for row in range(2):
        row_tiles = []
        for col in range(4):
            idx = row * 4 + col
            t = cv2.cvtColor(tiles[idx], cv2.COLOR_GRAY2BGR) if idx < len(tiles) else np.zeros((h, w, 3), np.uint8)
            if idx < len(labels) and labels[idx]:
                cv2.putText(t, labels[idx], (6, int(30*scale)),
                            font, 0.7*scale, (0, 0, 0),   2, cv2.LINE_AA)
                cv2.putText(t, labels[idx], (4, int(28*scale)),
                            font, 0.7*scale, (0, 255, 255), 1, cv2.LINE_AA)
            row_tiles.append(t)
        rows.append(np.hstack(row_tiles))
    panel = np.vstack(rows)
    # Scale down so it's not enormous
    panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
    name  = os.path.basename(filepath)
    out   = os.path.join(LABELED_DIR, "panel_" + os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(out, panel)


def process_image(filepath, save_panel=False):
    img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    fov_mask, (cx, cy), r = get_fov_mask(img)
    scratch_mask, texture, scratch_area_px, qc_flag, reps = detect_scratch(img, fov_mask, cx, cy, r)
    width_um, width_std_um = measure_width(scratch_mask)
    annotate_and_save(filepath, scratch_mask, fov_mask,
                      scratch_area_px, width_um, width_std_um, qc_flag)
    if save_panel:
        save_debug_panel(filepath, reps, scratch_mask, fov_mask)
    return scratch_area_px, width_um, width_std_um, qc_flag


def main():
    image_files = sorted(
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    print(f"Processing {len(image_files)} images...\n")
    rows = []
    for i, fp in enumerate(image_files):
        name = os.path.basename(fp)
        area, width, std, qc = process_image(fp, save_panel=(i == 0))
        w_str = f"{width:.0f}" if (width and np.isfinite(width)) else "nan"
        print(f"  {name:35s}  area={area:>9,} px   width={w_str} um   qc={qc}")
        rows.append({"file": name, "scratch_area_px": area,
                     "scratch_width_um": round(width, 1) if (width and np.isfinite(width)) else None,
                     "scratch_width_std_um": round(std, 1) if (std and np.isfinite(std)) else None,
                     "qc_flag": qc})

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","scratch_area_px",
                                               "scratch_width_um","scratch_width_std_um","qc_flag"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.")
    print(f"  Labeled images → {LABELED_DIR}")
    print(f"  Results CSV    → {CSV_PATH}")


if __name__ == "__main__":
    main()
