
import argparse
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

OUT_DIR = Path(__file__).parent / "final_output"
DEBUG_DIR = OUT_DIR / "debug"
OUT_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

UM_PER_PX = 840 / 557
TIME_LABEL = {"10": "10 s", "1m": "1 min", "5m": "5 min", "10m": "10 min"}
TIME_ORDER = ["10 s", "1 min", "5 min", "10 min"]
LIGHT_ORDER = ["control", "R", "UV", "unknown"]
LIGHT_LABEL = {"control": "Control", "R": "Red light", "UV": "UV light", "unknown": "Unknown"}


def parse_treatment_filename(name):
    stem = Path(name).stem
    m = re.match(r"t([A-E])(\d+)_control_([A-Z]+)_(\w+)", stem)
    if m:
        return {
            "row": m[1],
            "col": int(m[2]),
            "well": f"{m[1]}{int(m[2])}",
            "light": "control",
            "time": TIME_LABEL.get(m[4], m[4]),
        }
    m = re.match(r"t([A-E])(\d+)_([A-Z]+)_(\w+)", stem)
    if m:
        return {
            "row": m[1],
            "col": int(m[2]),
            "well": f"{m[1]}{int(m[2])}",
            "light": m[3],
            "time": TIME_LABEL.get(m[4], m[4]),
        }
    return None


def extract_well(name):
    m = re.search(r"([A-E]\d+)", Path(name).stem)
    return m.group(1) if m else None


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


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


def preprocess_image(img, fov_mask):
    img = img.astype(np.float32)
    bg = cv2.GaussianBlur(img, (0, 0), 61)
    flat = img - bg
    vals = flat[fov_mask == 255]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    flat = np.clip((flat - lo) / (hi - lo + 1e-6), 0, 1)
    return flat.astype(np.float32)


def compute_texture_map(img, mask, block_size=31):
    mask_f = (mask == 255).astype(np.float32)
    img_m = img * mask_f
    local_sum = uniform_filter(img_m, size=block_size)
    local_cnt = np.maximum(uniform_filter(mask_f, size=block_size), 1e-6)
    local_mean = local_sum / local_cnt
    local_sq = uniform_filter(img_m ** 2, size=block_size)
    local_var = np.maximum(local_sq / local_cnt - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = cv2.GaussianBlur(grad, (0, 0), 3)
    tex = local_std + 0.35 * grad
    tex[mask == 0] = 0
    vals = tex[mask == 255]
    if len(vals) == 0:
        return np.zeros_like(tex, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    tex = np.clip((tex - lo) / (hi - lo + 1e-6), 0, 1)
    tex[mask == 0] = 0
    return tex.astype(np.float32)


def classify_cells(img, scratch_mask, fov_mask):
    flat = preprocess_image(img, fov_mask)
    texture = compute_texture_map(flat, fov_mask, block_size=21)
    non_scratch = (fov_mask == 255) & (scratch_mask == 0)
    if non_scratch.sum() == 0:
        return np.zeros_like(fov_mask, dtype=np.uint8), np.nan, texture
    vals = texture[non_scratch]
    thresh = np.percentile(vals, 55)
    cell_mask = (texture >= thresh) & non_scratch
    cell_mask = cv2.morphologyEx(
        cell_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    cell_mask = cv2.morphologyEx(
        cell_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    confluence = float((cell_mask > 0).sum()) / float(non_scratch.sum())
    return cell_mask, confluence, texture


def _common_detection_maps(img, mask, cx, cy, r, band_frac):
    flat = preprocess_image(img, mask)
    texture = compute_texture_map(flat, mask, block_size=31)

    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), int(r * 0.78), 255, -1)

    center_band = np.zeros_like(mask)
    band_half_width = int(r * band_frac)
    x0 = max(0, cx - band_half_width)
    x1 = min(mask.shape[1], cx + band_half_width)
    center_band[:, x0:x1] = 255

    valid = (mask == 255) & (inner == 255) & (center_band == 255)
    scratch_score = 1.0 - texture
    scratch_score = cv2.GaussianBlur(scratch_score, (0, 0), 7)
    return flat, texture, inner, center_band, valid, scratch_score


def _score_component(labels, stats, cents, i, cx, cy, r, valid, center_band):
    x, y, w, h, area = stats[i]
    c_x, c_y = cents[i]
    if area < 3500:
        return None

    comp = (labels == i).astype(np.uint8)
    overlap_center = int(np.sum((comp > 0) & (center_band > 0)))
    overlap_valid = int(np.sum((comp > 0) & valid))
    if overlap_center < 1200 or overlap_valid < 1200:
        return None

    ys, xs = np.where(comp > 0)
    if len(xs) == 0:
        return None

    min_x = int(xs.min())
    max_x = int(xs.max())
    touches_left_outer = min_x < int(cx - r * 0.55)
    touches_right_outer = max_x > int(cx + r * 0.55)

    edge_penalty = 0
    if touches_left_outer:
        edge_penalty += 8000
    if touches_right_outer:
        edge_penalty += 8000

    aspect = h / max(w, 1)
    score = (
        1.5 * area
        + 4.0 * overlap_center
        + 1000.0 * aspect
        - 2.5 * abs(c_x - cx)
        - 0.5 * abs(c_y - cy)
        - edge_penalty
    )
    return score


def detect_scratch_before(img, mask, cx, cy, r):
    flat, texture, inner, center_band, valid, scratch_score = _common_detection_maps(img, mask, cx, cy, r, band_frac=0.45)
    vals = scratch_score[valid]
    if len(vals) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "bad_fov"

    thresh = np.percentile(vals, 66)
    cand = ((scratch_score >= thresh) & valid).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71)))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    best_label = None
    best_score = -1e18

    for i in range(1, n):
        score = _score_component(labels, stats, cents, i, cx, cy, r, valid, center_band)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_component"

    scratch_core = (labels == best_label).astype(np.uint8) * 255
    scratch_core = cv2.bitwise_and(scratch_core, scratch_core, mask=center_band.astype(np.uint8))
    scratch_core = cv2.dilate(scratch_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    scratch_core = cv2.bitwise_and(scratch_core, scratch_core, mask=inner.astype(np.uint8))

    expand_thresh = np.percentile(scratch_score[valid], 50)
    expand_candidates = ((scratch_score >= expand_thresh) & (mask == 255) & (inner == 255)).astype(np.uint8) * 255
    dilated = cv2.dilate(scratch_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    expanded = ((dilated > 0) & (expand_candidates > 0)).astype(np.uint8) * 255

    scratch_mask = cv2.bitwise_or(scratch_core, expanded)
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81)))
    scratch_mask[mask == 0] = 0
    return finalize_width_metrics(scratch_mask, texture, mask)


def detect_scratch_after(img, mask, cx, cy, r):
    flat, texture, inner, center_band, valid, scratch_score = _common_detection_maps(img, mask, cx, cy, r, band_frac=0.40)
    vals = scratch_score[valid]
    if len(vals) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "bad_fov"

    thresh = np.percentile(vals, 72)
    cand = ((scratch_score >= thresh) & valid).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    best_label = None
    best_score = -1e18

    for i in range(1, n):
        score = _score_component(labels, stats, cents, i, cx, cy, r, valid, center_band)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    scratch_core = (labels == best_label).astype(np.uint8) * 255
    scratch_core = cv2.bitwise_and(scratch_core, scratch_core, mask=center_band.astype(np.uint8))
    scratch_core = cv2.bitwise_and(scratch_core, scratch_core, mask=inner.astype(np.uint8))

    ys, xs = np.where(scratch_core > 0)
    if len(xs) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    core_area = int((scratch_core > 0).sum())
    if core_area < 900:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    inside_vals = texture[scratch_core > 0]
    if len(inside_vals) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    border_ring = cv2.dilate(scratch_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    border_ring = ((border_ring > 0) & (scratch_core == 0) & (mask == 255) & (inner == 255)).astype(np.uint8) * 255
    border_vals = texture[border_ring > 0]
    if len(border_vals) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    contrast = float(np.median(border_vals) - np.median(inside_vals))
    if contrast < 0.0035:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    expand_thresh = np.percentile(scratch_score[valid], 52)
    expand_candidates = ((scratch_score >= expand_thresh) & (mask == 255) & (inner == 255) & (center_band == 255)).astype(np.uint8) * 255
    dilated = cv2.dilate(scratch_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    expanded = ((dilated > 0) & (expand_candidates > 0)).astype(np.uint8) * 255

    scratch_mask = cv2.bitwise_or(scratch_core, expanded)
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    scratch_mask[mask == 0] = 0

    ys2, xs2 = np.where(scratch_mask > 0)
    if len(xs2) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    width_span = int(xs2.max() - xs2.min() + 1)
    center_band_width = int(np.sum(center_band[center_band.shape[0] // 2] > 0))
    if width_span > 0.9 * max(center_band_width, 1):
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, "no_scratch"

    return finalize_width_metrics(scratch_mask, texture, mask)


def finalize_width_metrics(scratch_mask, texture, mask):
    ys = np.where(np.any(scratch_mask > 0, axis=1))[0]
    widths = []
    for y in ys:
        xs = np.where(scratch_mask[y] > 0)[0]
        if len(xs) < 10:
            continue
        widths.append(xs[-1] - xs[0] + 1)

    scratch_area_px = int((scratch_mask > 0).sum())
    if len(widths) < 20:
        return scratch_mask, texture, np.nan, np.nan, scratch_area_px, np.nan, len(widths), "too_few_rows"

    widths = np.array(widths, dtype=np.float32)
    med = np.median(widths)
    keep = (widths > 0.4 * med) & (widths < 2.2 * med)
    widths_kept = widths[keep]
    valid_rows = int(len(widths_kept))

    if len(widths_kept) < 10:
        return scratch_mask, texture, np.nan, np.nan, scratch_area_px, np.nan, valid_rows, "width_outliers"

    width_um = float(np.mean(widths_kept) * UM_PER_PX)
    width_std_um = float(np.std(widths_kept) * UM_PER_PX)
    width_cv = float(np.std(widths_kept) / (np.mean(widths_kept) + 1e-6))

    qc_flag = "ok"
    if valid_rows < 20:
        qc_flag = "low_rows"
    if width_cv > 0.35:
        qc_flag = "high_width_cv"

    return scratch_mask, texture, width_um, width_std_um, scratch_area_px, width_cv, valid_rows, qc_flag


def save_debug_overlay(path_in, scratch_mask, fov_mask, img, out_path, scale=4):
    raw = cv2.imread(str(path_in), cv2.IMREAD_GRAYSCALE)
    h, w = raw.shape
    s = scale
    small = cv2.resize(raw, (w // s, h // s))
    overlay = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR).astype(np.float32)
    sm = cv2.resize(scratch_mask, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
    fm = cv2.resize(fov_mask, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
    cell_mask, _, _ = classify_cells(img, scratch_mask, fov_mask)
    cb = cv2.resize(cell_mask, (w // s, h // s), interpolation=cv2.INTER_NEAREST)
    outside = fm <= 127
    overlay[outside] *= 0.15
    cell_px = (cb > 127) & (sm <= 127) & (fm > 127)
    overlay[cell_px] = overlay[cell_px] * 0.5 + np.array([0, 150, 0], dtype=np.float32) * 0.5
    scratch_px = (sm > 127) & (fm > 127)
    overlay[scratch_px] = overlay[scratch_px] * 0.35 + np.array([40, 40, 220], dtype=np.float32) * 0.65
    edge = cv2.morphologyEx((sm > 127).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    overlay[edge > 0] = np.array([0, 255, 255], dtype=np.float32)
    cv2.imwrite(str(out_path), overlay.astype(np.uint8))


def build_panel(path_in, out_path, mode):
    img = load_gray(path_in)
    fov_mask, (cx, cy), radius = get_fov_mask(img)
    if mode == "after":
        scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch_after(img, fov_mask, cx, cy, radius)
    else:
        scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch_before(img, fov_mask, cx, cy, radius)
    cell_mask, confluence, _ = classify_cells(img, scratch_mask, fov_mask)
    raw = cv2.imread(str(path_in), cv2.IMREAD_GRAYSCALE)
    flat = preprocess_image(img, fov_mask)

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.imshow(raw, cmap="gray")
    ax1.set_title("Raw image")
    ax1.axis("off")

    ax2.imshow(flat, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Flat-field corrected")
    ax2.axis("off")

    ax3.imshow(texture, cmap="viridis", vmin=0, vmax=1)
    ax3.set_title("Texture map")
    ax3.axis("off")

    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB).astype(np.float32)
    overlay[fov_mask == 0] *= 0.15
    overlay[(cell_mask > 0) & (scratch_mask == 0)] = overlay[(cell_mask > 0) & (scratch_mask == 0)] * 0.5 + np.array([0, 150, 0], dtype=np.float32) * 0.5
    overlay[scratch_mask > 0] = overlay[scratch_mask > 0] * 0.35 + np.array([40, 40, 220], dtype=np.float32) * 0.65
    ax4.imshow(np.clip(overlay, 0, 255).astype(np.uint8))

    if np.isfinite(width_mean):
        title = f"{mode} | width={width_mean:.0f} µm | conf={confluence:.3f} | rows={valid_rows} | qc={qc_flag}"
    else:
        title = f"{mode} | width=nan | conf={confluence:.3f} | rows={valid_rows} | qc={qc_flag}"
    ax4.set_title(title)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def process_one_image(path, mode, prefix):
    fpath = Path(path)
    img = load_gray(fpath)
    fov_mask, (cx, cy), radius = get_fov_mask(img)
    if mode == "after":
        scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch_after(img, fov_mask, cx, cy, radius)
    else:
        scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch_before(img, fov_mask, cx, cy, radius)
    cell_mask, confluence, _ = classify_cells(img, scratch_mask, fov_mask)
    overlay_path = DEBUG_DIR / f"{prefix}_{fpath.stem}_overlay.jpg"
    panel_path = DEBUG_DIR / f"{prefix}_{fpath.stem}_panel.png"
    save_debug_overlay(fpath, scratch_mask, fov_mask, img, overlay_path)
    build_panel(fpath, panel_path, mode)
    return {
        "file": fpath.name,
        "well": extract_well(fpath.name),
        "width_um": width_mean,
        "width_std_um": width_std,
        "scratch_area_px": scratch_area_px,
        "width_cv": width_cv,
        "valid_rows": valid_rows,
        "cell_confluence_outside_scratch": confluence,
        "qc_flag": qc_flag,
        "overlay_path": str(overlay_path),
        "panel_path": str(panel_path),
        "mode": mode,
    }


def test_images(mode, paths):
    records = []
    for image_path in paths:
        rec = process_one_image(image_path, mode=mode, prefix=mode)
        records.append(rec)
        print(f"\nTesting ({mode}): {rec['file']}")
        if np.isfinite(rec["width_um"]):
            print(f"  Width:             {rec['width_um']:.0f} ± {rec['width_std_um']:.0f} µm")
        else:
            print("  Width:             nan")
        print(f"  Scratch area:      {rec['scratch_area_px']} px")
        if np.isfinite(rec["width_cv"]):
            print(f"  Width CV:          {rec['width_cv']:.3f}")
        else:
            print("  Width CV:          nan")
        print(f"  Valid width rows:  {rec['valid_rows']}")
        print(f"  Cell confluence:   {rec['cell_confluence_outside_scratch']:.3f}")
        print(f"  QC flag:           {rec['qc_flag']}")
        print(f"  Overlay -> {rec['overlay_path']}")
        print(f"  Panel   -> {rec['panel_path']}")
    pd.DataFrame(records).to_csv(OUT_DIR / f"test_results_{mode}.csv", index=False)
    print(f"\nSaved test summary -> {OUT_DIR / f'test_results_{mode}.csv'}")


def build_treatment_lookup(before_dir):
    rows = []
    for f in sorted(Path(before_dir).glob("t*.jpg")):
        meta = parse_treatment_filename(f.name)
        if meta is not None:
            rows.append({"well": meta["well"], "light": meta["light"], "time": meta["time"], "treatment_file": f.name})
    if not rows:
        return pd.DataFrame(columns=["well", "light", "time", "treatment_file"])
    lookup = pd.DataFrame(rows).drop_duplicates(subset=["well"], keep="first")
    return lookup


def run_healing(before_dir, after_dir):
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)

    baseline_files = [f for f in sorted(before_dir.glob("*.jpg")) if not f.name.startswith("t")]
    after_files = sorted(after_dir.glob("*.jpg"))

    before_records = [process_one_image(f, mode="before", prefix="before") for f in baseline_files if extract_well(f.name) is not None]
    after_records = [process_one_image(f, mode="after", prefix="after") for f in after_files if extract_well(f.name) is not None]

    before_df = pd.DataFrame(before_records)
    after_df = pd.DataFrame(after_records)
    treatment_df = build_treatment_lookup(before_dir)

    before_df.to_csv(OUT_DIR / "before_segmented.csv", index=False)
    after_df.to_csv(OUT_DIR / "after_segmented.csv", index=False)

    before_ok = before_df[before_df["qc_flag"] == "ok"].copy()
    after_use = after_df.copy()
    after_use["width_um_effective"] = after_use["width_um"]
    after_use.loc[after_use["qc_flag"] == "no_scratch", "width_um_effective"] = 0.0
    after_use = after_use[after_use["qc_flag"].isin(["ok", "no_scratch"])].copy()

    merged = before_ok.merge(after_use, on="well", suffixes=("_before", "_after"))
    if len(treatment_df):
        merged = merged.merge(treatment_df, on="well", how="left")
    else:
        merged["light"] = "unknown"
        merged["time"] = "unknown"
        merged["treatment_file"] = ""

    merged["light"] = merged["light"].fillna("unknown")
    merged["time"] = merged["time"].fillna("unknown")
    merged["closure_um"] = merged["width_um_before"] - merged["width_um_effective"]
    merged["percent_closure"] = merged["closure_um"] / merged["width_um_before"]
    merged.to_csv(OUT_DIR / "healing_results.csv", index=False)

    print("\nMatched wells with usable after QC:")
    if len(merged):
        cols = [
            "well",
            "light",
            "time",
            "file_before",
            "width_um_before",
            "file_after",
            "qc_flag_after",
            "width_um_after",
            "width_um_effective",
            "closure_um",
            "percent_closure",
        ]
        print(merged[cols].to_string(index=False))
    else:
        print("None")

    make_healing_graphs(merged)

    print("\nSaved:")
    print(OUT_DIR / "before_segmented.csv")
    print(OUT_DIR / "after_segmented.csv")
    print(OUT_DIR / "healing_results.csv")
    print(DEBUG_DIR)


def make_healing_graphs(df):
    if df.empty:
        print("No matched data for healing graphs.")
        return

    df = df.copy()
    df = df.sort_values("percent_closure")

    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    plt.scatter(x, df["width_um_before"], s=45, label="Before")
    plt.scatter(x, df["width_um_effective"], s=45, label="After")
    for i, row in enumerate(df.itertuples()):
        plt.plot([i, i], [row.width_um_before, row.width_um_effective], alpha=0.6)
    plt.xticks(x, df["well"], rotation=90)
    plt.ylabel("Scratch width (µm)")
    plt.title("Paired before vs after scratch width by well")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "healing_paired_before_after.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    vals = df["percent_closure"].dropna()
    plt.hist(vals, bins=12)
    plt.xlabel("Percent closure")
    plt.ylabel("Count")
    plt.title("Distribution of 24h percent closure")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "healing_percent_closure_hist.png", dpi=180)
    plt.close()

    if "light" in df.columns:
        plot_df = df[df["light"].isin(["control", "R", "UV"])].copy()
        if len(plot_df):
            order = ["control", "R", "UV"]
            groups = [plot_df.loc[plot_df["light"] == light, "percent_closure"].dropna().values for light in order]
            labels = [LIGHT_LABEL[light] for light in order]
            nonempty = [(g, l) for g, l in zip(groups, labels) if len(g)]
            if len(nonempty):
                groups2, labels2 = zip(*nonempty)
                plt.figure(figsize=(7, 5))
                plt.boxplot(groups2, tick_labels=labels2)
                plt.ylabel("Percent closure")
                plt.title("Percent closure by treatment")
                plt.tight_layout()
                plt.savefig(OUT_DIR / "healing_percent_closure_by_treatment.png", dpi=180)
                plt.close()

                summary = (
                    plot_df.groupby("light")
                    .agg(
                        n=("well", "count"),
                        mean_percent_closure=("percent_closure", "mean"),
                        sem_percent_closure=("percent_closure", "sem"),
                        mean_closure_um=("closure_um", "mean"),
                        sem_closure_um=("closure_um", "sem"),
                    )
                    .reset_index()
                )
                summary.to_csv(OUT_DIR / "healing_summary_by_treatment.csv", index=False)
                print("\nHealing summary by treatment:")
                print(summary.to_string(index=False))

    summary_all = df[["closure_um", "percent_closure"]].agg(["count", "mean", "std", "median", "min", "max"]).round(4)
    summary_all.to_csv(OUT_DIR / "healing_overall_summary.csv")
    print("\nOverall healing summary:")
    print(summary_all.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_before", nargs="+", help="test one or more before images")
    parser.add_argument("--test_after", nargs="+", help="test one or more after images")
    parser.add_argument("--healing", action="store_true", help="run before/after healing analysis")
    parser.add_argument("--before_dir", default="before_images", help="directory of before images")
    parser.add_argument("--after_dir", default="after_images", help="directory of after images")
    args = parser.parse_args()

    if args.test_before:
        test_images("before", args.test_before)
        return

    if args.test_after:
        test_images("after", args.test_after)
        return

    if args.healing:
        run_healing(args.before_dir, args.after_dir)
        return

    print("Use one of: --test_before img1.jpg [img2.jpg ...] | --test_after img1.jpg [img2.jpg ...] | --healing")


if __name__ == "__main__":
    main()
