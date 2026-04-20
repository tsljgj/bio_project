"""
Scratch Assay Image Analysis
=============================
Measures scratch width and cell coverage for each treatment well at t=0.
Designed to be extended with 24h images once available.

Usage:
    python3 analysis.py
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import gaussian_filter

# ─── Paths ────────────────────────────────────────────────────────────────────
VIK_DIR   = Path(__file__).parent / "Vik"
OUT_DIR   = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "debug").mkdir(exist_ok=True)

# ─── Calibration ──────────────────────────────────────────────────────────────
UM_PER_PX = 840 / 557   # ~1.508 µm per pixel  (840 µm scale bar = 557 px)

# ─── Condition parsing ────────────────────────────────────────────────────────
TIME_LABEL = {"10": "10 s", "1m": "1 min", "5m": "5 min", "10m": "10 min"}
TIME_ORDER = ["10 s", "1 min", "5 min", "10 min"]
COND_COLOR = {"UV": "#9B2335", "R": "#2166AC", "control": "#555555"}
COND_LABEL = {"UV": "UV light", "R": "Red light", "control": "No-light control"}

def parse_filename(name):
    """
    Parse a t-prefix filename into its components.
    Examples:
      tA9_UV_10   → {row:A, col:9, light:UV,      time:10 s}
      tB1_control_R_10 → {row:B, col:1, light:control, time:10 s}
      tC4_R_1m    → {row:C, col:4, light:R,       time:1 min}
    """
    stem = Path(name).stem  # strip .jpg
    # Control wells: tX1_control_<light>_<time>
    m = re.match(r"t([A-E])(\d+)_control_([A-Z]+)_(\w+)", stem)
    if m:
        return dict(row=m[1], col=int(m[2]), light="control",
                    time=TIME_LABEL.get(m[4], m[4]))
    # Treatment wells: tX<col>_<light>_<time>
    m = re.match(r"t([A-E])(\d+)_([A-Z]+)_(\w+)", stem)
    if m:
        return dict(row=m[1], col=int(m[2]), light=m[3],
                    time=TIME_LABEL.get(m[4], m[4]))
    return None


# ─── Image utilities ──────────────────────────────────────────────────────────

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def get_fov_mask(img):
    """Return binary mask of the circular field-of-view."""
    h, w = img.shape
    # threshold out the black background
    _, rough = cv2.threshold(img.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * 0.97), 255, -1)  # 3% inset for edge artefacts
    return mask, (int(cx), int(cy)), int(r)


def correct_illumination(img, mask, sigma=300):
    """
    Subtract a large-scale background to flatten uneven illumination.
    Uses a heavily blurred version of the image as the background estimate.
    """
    bg = gaussian_filter(img, sigma=sigma)
    corrected = img - bg
    # Normalise to [0, 255] inside the mask
    roi = corrected[mask == 255]
    corrected = (corrected - roi.min()) / (roi.max() - roi.min()) * 255
    corrected[mask == 0] = 0
    return corrected.astype(np.float32)


def detect_scratch(img_corrected, mask):
    """
    Detect the scratch (cell-free stripe) using a center-seeded approach.

    Key insight: the microscopist always aims the objective at the scratch,
    so the scratch ALWAYS passes through (or very near) the image center.
    Large dead-cell zones caused by UV damage are also bright but tend to
    be blob-shaped and off-center — they are rejected by the aspect-ratio
    and center-overlap criteria.

    Steps:
      1. Heavy Gaussian blur to suppress individual cell texture.
      2. Threshold to find candidate bright (cell-free) regions.
      3. For each connected component, compute:
           score = center_overlap_fraction * aspect_ratio
         where aspect_ratio comes from PCA eigenvalue ratio.
      4. The highest-scoring component is the scratch.
    """
    h, w = img_corrected.shape

    # 1. Suppress cell texture with a large blur
    smooth = gaussian_filter(img_corrected, sigma=80)
    smooth[mask == 0] = 0

    # 2. Threshold: keep top 35% brightest pixels inside FOV
    thresh = np.percentile(smooth[mask == 255], 65)
    bright = ((smooth >= thresh) & (mask == 255)).astype(np.uint8) * 255

    # Morphological close to bridge small gaps within the scratch
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    # 3. Score each connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n_labels < 2:
        return bright, smooth

    # Center circle mask (inner 30% of FOV radius)
    cy, cx = h // 2, w // 2
    r_center = int(min(h, w) * 0.30)
    yy, xx = np.mgrid[0:h, 0:w]
    center_circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < r_center ** 2

    min_area = h * w * 0.005   # ignore components < 0.5% of image

    best_label, best_score = 1, -1
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue

        comp = labels == i
        # center overlap: fraction of the component inside the center circle
        overlap = np.sum(comp & center_circle) / (np.sum(comp) + 1e-6)

        # aspect ratio via PCA
        ys_c, xs_c = np.where(comp)
        pts = np.column_stack([xs_c, ys_c]).astype(float)
        cov = np.cov(pts.T)
        eigvals = np.linalg.eigvalsh(cov)
        aspect = eigvals.max() / (eigvals.min() + 1e-6)

        score = overlap * aspect
        if score > best_score:
            best_score = score
            best_label = i

    scratch_mask = (labels == best_label).astype(np.uint8) * 255
    return scratch_mask, smooth


def measure_scratch_width(scratch_mask, img_corrected, n_samples=40):
    """
    Sample the scratch width (in pixels) at `n_samples` positions along its
    length by measuring the run-length of the mask in the perpendicular
    direction.

    Returns mean and std width in µm.
    """
    h, w = scratch_mask.shape

    # Fit a line to the scratch centroid positions row-by-row
    ys, xs = np.where(scratch_mask == 255)
    if len(ys) < 100:
        return np.nan, np.nan

    # Determine primary axis of scratch (PCA on the centroid points)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean_pt = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Major axis = eigenvector with largest eigenvalue
    major = eigvecs[:, np.argmax(eigvals)]   # direction along scratch
    minor = eigvecs[:, np.argmin(eigvals)]   # perpendicular (= width direction)

    # Sample positions along the major axis
    t_vals = (pts - mean_pt) @ major
    t_min, t_max = t_vals.min(), t_vals.max()
    sample_ts = np.linspace(t_min * 0.8, t_max * 0.8, n_samples)

    widths = []
    for t in sample_ts:
        # Centre point of this cross-section
        cx = int(mean_pt[0] + t * major[0])
        cy = int(mean_pt[1] + t * major[1])
        # Walk in the minor (width) direction until we leave the scratch
        hit = []
        for direction in [1, -1]:
            for step in range(1, 2000):
                px = int(cx + direction * step * minor[0])
                py = int(cy + direction * step * minor[1])
                if px < 0 or px >= w or py < 0 or py >= h:
                    break
                if scratch_mask[py, px] == 0:
                    hit.append(step)
                    break
        if len(hit) == 2:
            widths.append(sum(hit))

    widths = np.array(widths)
    widths_um = widths * UM_PER_PX
    return widths_um.mean(), widths_um.std()


def measure_cell_coverage(img_corrected, scratch_mask, fov_mask, edge_sigma=2):
    """
    Estimate cell coverage in the non-scratch region using edge density.
    More cell edges per unit area → higher coverage.
    Returns fraction of non-scratch FOV pixels that contain cell edges.
    """
    non_scratch = ((fov_mask == 255) & (scratch_mask == 0)).astype(np.uint8)
    if non_scratch.sum() == 0:
        return np.nan

    # Canny edges on the corrected image
    blurred = gaussian_filter(img_corrected, sigma=edge_sigma)
    blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
    edges = cv2.Canny(blurred_u8, threshold1=20, threshold2=60)

    edge_px   = int(np.sum((edges > 0) & (non_scratch > 0)))
    total_px  = int(non_scratch.sum())
    return edge_px / total_px


# ─── Debug overlay ────────────────────────────────────────────────────────────

def save_debug_overlay(path_in, scratch_mask, fov_mask, out_path, scale=4):
    """Save a small overlay image for visual QC."""
    img = cv2.imread(str(path_in), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = scale
    small = cv2.resize(img, (w // s, h // s))
    overlay = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    sm = cv2.resize(scratch_mask, (w // s, h // s))
    fm = cv2.resize(fov_mask,     (w // s, h // s))

    # Tint scratch region red
    overlay[sm > 127, 0] = 0
    overlay[sm > 127, 1] = 0
    overlay[sm > 127, 2] = min(255, overlay[sm > 127, 2].astype(int).mean() + 80)
    overlay[sm > 127] = [60, 60, 220]

    cv2.imwrite(str(out_path), overlay)


# ─── Main analysis loop ───────────────────────────────────────────────────────

def analyse_all():
    records = []
    t_files = sorted(VIK_DIR.glob("t*.jpg"))
    print(f"Found {len(t_files)} treatment/control images.")

    for fpath in t_files:
        meta = parse_filename(fpath.name)
        if meta is None:
            print(f"  [SKIP] Could not parse: {fpath.name}")
            continue

        print(f"  Processing {fpath.name} ...", end=" ", flush=True)
        img = load_gray(fpath)
        fov_mask, (cx, cy), radius = get_fov_mask(img)
        corrected = correct_illumination(img, fov_mask, sigma=400)
        scratch_mask, blurred = detect_scratch(corrected, fov_mask)
        width_mean, width_std = measure_scratch_width(scratch_mask, corrected)
        coverage = measure_cell_coverage(corrected, scratch_mask, fov_mask)

        debug_path = OUT_DIR / "debug" / (fpath.stem + "_overlay.jpg")
        save_debug_overlay(fpath, scratch_mask, fov_mask, debug_path)

        rec = {**meta, "file": fpath.name,
               "scratch_width_um": width_mean, "scratch_width_std_um": width_std,
               "cell_edge_density": coverage}
        records.append(rec)
        print(f"width={width_mean:.0f} µm  edge_density={coverage:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "scratch_measurements.csv", index=False)
    return df


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scratch Assay – t = 0 (immediately post-exposure)", fontsize=13)

    for ax, metric, ylabel, title in [
        (axes[0], "scratch_width_um",   "Scratch width (µm)",
         "Scratch Width by Condition"),
        (axes[1], "cell_edge_density",  "Edge density (edges / px)",
         "Cell Coverage Outside Scratch"),
    ]:
        for light, grp in df.groupby("light"):
            if light == "control":
                continue
            times  = []
            means  = []
            sems   = []
            for t in TIME_ORDER:
                sub = grp[grp["time"] == t][metric].dropna()
                if len(sub):
                    times.append(t)
                    means.append(sub.mean())
                    sems.append(sub.sem())

            color = COND_COLOR.get(light, "gray")
            label = COND_LABEL.get(light, light)
            ax.errorbar(times, means, yerr=sems, marker="o", linewidth=2,
                        markersize=7, capsize=4, color=color, label=label)

        # Controls (pooled across time-matched groups)
        ctrl_vals = df[df["light"] == "control"][metric].dropna()
        if len(ctrl_vals):
            ax.axhline(ctrl_vals.mean(), color=COND_COLOR["control"],
                       linestyle="--", linewidth=1.5, label="No-light control (mean)")
            ax.axhspan(ctrl_vals.mean() - ctrl_vals.sem(),
                       ctrl_vals.mean() + ctrl_vals.sem(),
                       color=COND_COLOR["control"], alpha=0.12)

        ax.set_xlabel("Light exposure duration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "scratch_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")


def plot_replicate_scatter(df):
    """Dot plot showing individual replicates (rows A–E) for each condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Individual Replicates – t = 0", fontsize=13)

    for ax, metric, ylabel in [
        (axes[0], "scratch_width_um",  "Scratch width (µm)"),
        (axes[1], "cell_edge_density", "Edge density"),
    ]:
        x_positions = {}
        x_tick_labels = []
        x = 0
        for light in ["R", "UV"]:
            sub_df = df[df["light"] == light]
            for t in TIME_ORDER:
                label = f"{COND_LABEL[light]}\n{t}"
                x_positions[(light, t)] = x
                x_tick_labels.append(label)
                vals = sub_df[sub_df["time"] == t][metric].dropna()
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax.scatter(x + jitter, vals, color=COND_COLOR[light],
                           alpha=0.7, s=50, zorder=3)
                if len(vals):
                    ax.plot([x - 0.25, x + 0.25], [vals.mean(), vals.mean()],
                            color=COND_COLOR[light], linewidth=2.5, zorder=4)
                x += 1
            x += 0.5  # gap between UV/Red groups

        ax.set_xticks(list(range(len(x_tick_labels))))
        ax.set_xticklabels(x_tick_labels, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

        patches = [mpatches.Patch(color=COND_COLOR[l], label=COND_LABEL[l])
                   for l in ["R", "UV"]]
        ax.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "replicate_scatter.png"
    plt.savefig(out, dpi=150)
    print(f"Replicate scatter saved → {out}")


def print_summary(df):
    print("\n──────────────── Summary (mean ± SEM across replicates) ────────────────")
    summary = (df.groupby(["light", "time"])[["scratch_width_um", "cell_edge_density"]]
                 .agg(["mean", "sem"])
                 .round(2))
    print(summary.to_string())
    print("─────────────────────────────────────────────────────────────────────────")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = analyse_all()
    print_summary(df)
    plot_results(df)
    plot_replicate_scatter(df)
    print("\nDone. Check the output/ folder for results.")
