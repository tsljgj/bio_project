import os
import re
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, uniform_filter

OUT_DIR = Path(__file__).parent / 'final_output'
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / 'debug').mkdir(exist_ok=True)

UM_PER_PX = 840 / 557
TIME_LABEL = {'10': '10 s', '1m': '1 min', '5m': '5 min', '10m': '10 min'}
TIME_ORDER = ['10 s', '1 min', '5 min', '10 min']
COND_COLOR = {'UV': '#9B2335', 'R': '#2166AC', 'control': '#555555'}
COND_LABEL = {'UV': 'UV light', 'R': 'Red light', 'control': 'No-light control'}


def parse_filename(name):
    stem = Path(name).stem
    m = re.match(r't([A-E])(\d+)_control_([A-Z]+)_(\w+)', stem)
    if m:
        return dict(row=m[1], col=int(m[2]), light='control', time=TIME_LABEL.get(m[4], m[4]))
    m = re.match(r't([A-E])(\d+)_([A-Z]+)_(\w+)', stem)
    if m:
        return dict(row=m[1], col=int(m[2]), light=m[3], time=TIME_LABEL.get(m[4], m[4]))
    return None


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
    non_scratch = ((fov_mask == 255) & (scratch_mask == 0))
    if non_scratch.sum() == 0:
        return np.zeros_like(fov_mask, dtype=np.uint8), np.nan, texture
    vals = texture[non_scratch]
    thresh = np.percentile(vals, 55)
    cell_mask = (texture >= thresh) & non_scratch
    cell_mask = cv2.morphologyEx(cell_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    confluence = float((cell_mask > 0).sum()) / float(non_scratch.sum())
    return cell_mask, confluence, texture


def detect_scratch(img, mask, cx, cy, r):
    flat = preprocess_image(img, mask)
    texture = compute_texture_map(flat, mask, block_size=31)
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), int(r * 0.78), 255, -1)
    valid = (mask == 255) & (inner == 255)
    scratch_score = 1.0 - texture
    scratch_score = cv2.GaussianBlur(scratch_score, (0, 0), 7)
    vals = scratch_score[valid]
    if len(vals) == 0:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, 'bad_fov'
    thresh = np.percentile(vals, 75)
    cand = ((scratch_score >= thresh) & valid).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    best_label = None
    best_score = -1e18
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        c_x, c_y = cents[i]
        if area < 5000:
            continue
        score = area + 2.5 * h - 1.2 * abs(c_x - cx) - 0.3 * abs(c_y - cy)
        if score > best_score:
            best_score = score
            best_label = i
    if best_label is None:
        return np.zeros_like(mask), texture, np.nan, np.nan, 0, np.nan, 0, 'no_component'
    scratch_mask = (labels == best_label).astype(np.uint8) * 255
    # ─── EXPANSION STEP (fix missing edges) ───
    scratch_core = scratch_mask.copy()

    # allow slightly higher texture than core
    expand_thresh = np.percentile(texture[valid], 60)

    expand_candidates = ((texture < expand_thresh) & valid).astype(np.uint8) * 255

    # dilate core to connect nearby candidates
    dilated = cv2.dilate(scratch_core,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))

    expanded = ((dilated > 0) & (expand_candidates > 0)).astype(np.uint8) * 255

    # merge
    scratch_mask = cv2.bitwise_or(scratch_core, expanded)
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81)))
    scratch_mask[mask == 0] = 0
    ys = np.where(np.any(scratch_mask > 0, axis=1))[0]
    rows = []
    widths = []
    for y in ys:
        xs = np.where(scratch_mask[y] > 0)[0]
        if len(xs) < 10:
            continue
        rows.append(y)
        widths.append(xs[-1] - xs[0] + 1)
    scratch_area_px = int((scratch_mask > 0).sum())
    if len(widths) < 20:
        return scratch_mask, texture, np.nan, np.nan, scratch_area_px, np.nan, len(widths), 'too_few_rows'
    widths = np.array(widths, dtype=np.float32)
    med = np.median(widths)
    keep = (widths > 0.4 * med) & (widths < 2.2 * med)
    widths_kept = widths[keep]
    valid_rows = int(len(widths_kept))
    if len(widths_kept) < 10:
        return scratch_mask, texture, np.nan, np.nan, scratch_area_px, np.nan, valid_rows, 'width_outliers'
    width_um = float(np.mean(widths_kept) * UM_PER_PX)
    width_std_um = float(np.std(widths_kept) * UM_PER_PX)
    width_cv = float(np.std(widths_kept) / (np.mean(widths_kept) + 1e-6))
    qc_flag = 'ok'
    if valid_rows < 20:
        qc_flag = 'low_rows'
    if width_cv > 0.35:
        qc_flag = 'high_width_cv'
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
    edge_px = edge > 0
    overlay[edge_px] = np.array([0, 255, 255], dtype=np.float32)
    cv2.imwrite(str(out_path), overlay.astype(np.uint8))


def build_panel(path_in, out_path):
    img = load_gray(path_in)
    fov_mask, (cx, cy), radius = get_fov_mask(img)
    flat = preprocess_image(img, fov_mask)
    scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch(img, fov_mask, cx, cy, radius)
    cell_mask, confluence, _ = classify_cells(img, scratch_mask, fov_mask)
    raw = cv2.imread(str(path_in), cv2.IMREAD_GRAYSCALE)
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.imshow(raw, cmap='gray')
    ax1.set_title('Raw image')
    ax1.axis('off')
    ax2.imshow(flat, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Flat-field corrected')
    ax2.axis('off')
    ax3.imshow(texture, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('Texture map')
    ax3.axis('off')
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB).astype(np.float32)
    overlay[fov_mask == 0] *= 0.15
    overlay[(cell_mask > 0) & (scratch_mask == 0)] = overlay[(cell_mask > 0) & (scratch_mask == 0)] * 0.5 + np.array([0, 150, 0], dtype=np.float32) * 0.5
    overlay[scratch_mask > 0] = overlay[scratch_mask > 0] * 0.35 + np.array([40, 40, 220], dtype=np.float32) * 0.65
    ax4.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    title = f'Overlay | width={width_mean:.0f} µm | conf={confluence:.3f} | rows={valid_rows} | qc={qc_flag}' if np.isfinite(width_mean) else f'Overlay | width=nan | conf={confluence:.3f} | rows={valid_rows} | qc={qc_flag}'
    ax4.set_title(title)
    ax4.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def test_single(image_path):
    fpath = Path(image_path)
    if not fpath.exists():
        print(f'File not found: {fpath}')
        return
    print(f'Testing: {fpath.name}')
    img = load_gray(fpath)
    fov_mask, (cx, cy), radius = get_fov_mask(img)
    scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch(img, fov_mask, cx, cy, radius)
    cell_mask, confluence, _ = classify_cells(img, scratch_mask, fov_mask)
    debug_path = OUT_DIR / 'debug' / (fpath.stem + '_overlay.jpg')
    panel_path = OUT_DIR / 'debug' / (fpath.stem + '_panel.png')
    save_debug_overlay(fpath, scratch_mask, fov_mask, img, debug_path)
    build_panel(fpath, panel_path)
    print(f'  Width:             {width_mean:.0f} ± {width_std:.0f} µm' if np.isfinite(width_mean) else '  Width:             nan')
    print(f'  Scratch area:      {scratch_area_px} px')
    print(f'  Width CV:          {width_cv:.3f}' if np.isfinite(width_cv) else '  Width CV:          nan')
    print(f'  Valid width rows:  {valid_rows}')
    print(f'  Cell confluence:   {confluence:.3f}')
    print(f'  QC flag:           {qc_flag}')
    print(f'  Overlay -> {debug_path}')
    print(f'  Panel   -> {panel_path}')


def analyse_all(image_dir):
    image_dir = Path(image_dir)
    records = []
    t_files = sorted(image_dir.glob('t*.jpg'))
    print(f'Found {len(t_files)} treatment/control images in {image_dir}.')
    if len(t_files) == 0:
        print("No images found matching 't*.jpg'.")
        return pd.DataFrame()
    for fpath in t_files:
        meta = parse_filename(fpath.name)
        if meta is None:
            print(f'  [SKIP] Could not parse: {fpath.name}')
            continue
        print(f'  Processing {fpath.name} ...', end=' ', flush=True)
        img = load_gray(fpath)
        fov_mask, (cx, cy), radius = get_fov_mask(img)
        scratch_mask, texture, width_mean, width_std, scratch_area_px, width_cv, valid_rows, qc_flag = detect_scratch(img, fov_mask, cx, cy, radius)
        cell_mask, confluence, _ = classify_cells(img, scratch_mask, fov_mask)
        debug_path = OUT_DIR / 'debug' / (fpath.stem + '_overlay.jpg')
        save_debug_overlay(fpath, scratch_mask, fov_mask, img, debug_path)
        rec = {
            **meta,
            'file': fpath.name,
            'scratch_width_um': width_mean,
            'scratch_width_std_um': width_std,
            'scratch_area_px': scratch_area_px,
            'cell_confluence_outside_scratch': confluence,
            'valid_width_rows': valid_rows,
            'width_cv': width_cv,
            'qc_flag': qc_flag,
        }
        records.append(rec)
        w_text = f'{width_mean:.0f}' if np.isfinite(width_mean) else 'nan'
        print(f'width={w_text} µm confluence={confluence:.3f} qc={qc_flag}')
    df = pd.DataFrame(records)
    if len(df) > 0:
        df.to_csv(OUT_DIR / 'scratch_measurements.csv', index=False)
    return df


def plot_results(df):
    if df.empty or len(df[df['light'] != 'control']) == 0:
        print('Not enough treatment data to plot.')
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Scratch Assay — Measurements', fontsize=13)
    for ax, metric, ylabel, title in [
        (axes[0], 'scratch_width_um', 'Scratch width (µm)', 'Scratch Width by Condition'),
        (axes[1], 'cell_confluence_outside_scratch', 'Cell confluence outside scratch', 'Cell Confluence by Condition'),
    ]:
        for light, grp in df.groupby('light'):
            if light == 'control':
                continue
            times, means, sems = [], [], []
            for t in TIME_ORDER:
                sub = grp[(grp['time'] == t) & (grp['qc_flag'] != 'no_component')][metric].dropna()
                if len(sub):
                    times.append(t)
                    means.append(sub.mean())
                    sems.append(sub.sem() if len(sub) > 1 else 0)
            if not times:
                continue
            ax.errorbar(times, means, yerr=sems, marker='o', lw=2, ms=7, capsize=4, color=COND_COLOR.get(light, 'gray'), label=COND_LABEL.get(light, light))
        ctrl = df[df['light'] == 'control'][metric].dropna()
        if len(ctrl):
            m = ctrl.mean()
            se = ctrl.sem() if len(ctrl) > 1 else 0
            ax.axhline(m, color=COND_COLOR['control'], ls='--', lw=1.5, label='No-light control (mean)')
            ax.axhspan(m - se, m + se, color=COND_COLOR['control'], alpha=0.12)
        ax.set_xlabel('Light exposure duration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'scratch_analysis.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Plot saved -> {out}')


def plot_replicate_scatter(df):
    if df.empty or len(df[df['light'] != 'control']) == 0:
        print('Not enough data for replicate scatter.')
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Individual Replicates', fontsize=13)
    for ax, metric, ylabel in [
        (axes[0], 'scratch_width_um', 'Scratch width (µm)'),
        (axes[1], 'cell_confluence_outside_scratch', 'Cell confluence outside scratch'),
    ]:
        labels_list = []
        x = 0
        for light in ['R', 'UV']:
            sub_df = df[df['light'] == light]
            for t in TIME_ORDER:
                labels_list.append(f'{COND_LABEL[light]}\n{t}')
                vals = sub_df[sub_df['time'] == t][metric].dropna()
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax.scatter(x + jitter, vals, color=COND_COLOR[light], alpha=0.7, s=50, zorder=3)
                if len(vals):
                    ax.plot([x - 0.25, x + 0.25], [vals.mean(), vals.mean()], color=COND_COLOR[light], lw=2.5, zorder=4)
                x += 1
            x += 0.5
        ax.set_xticks(range(len(labels_list)))
        ax.set_xticklabels(labels_list, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(handles=[mpatches.Patch(color=COND_COLOR[l], label=COND_LABEL[l]) for l in ['R', 'UV']], fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / 'replicate_scatter.png'
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Replicate scatter saved -> {out}')


def print_summary(df):
    if df.empty:
        print('No data to summarize.')
        return
    print('\nSummary (mean ± SEM across replicates)')
    summary = (df.groupby(['light', 'time'])[['scratch_width_um', 'cell_confluence_outside_scratch']].agg(['mean', 'sem']).round(3))
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='+', help='one or more image paths to test')
    parser.add_argument('--dir', default=None, help='directory containing t*.jpg images for batch analysis')
    args = parser.parse_args()
    if args.test:
        for image_path in args.test:
            test_single(image_path)
        return
    if args.dir is None:
        print('Provide either --test image1.jpg [image2.jpg ...] or --dir /path/to/images')
        return
    df = analyse_all(args.dir)
    print_summary(df)
    plot_results(df)
    plot_replicate_scatter(df)
    print('\nDone. Check the output/ folder for results.')


if __name__ == '__main__':
    main()
