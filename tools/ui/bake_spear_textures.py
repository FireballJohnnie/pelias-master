#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Bakes the two spear layers used by the progress bar from the author's 100% reference image.

Input : pelias-spear-100.png (the finished stone spear, 2755 x 1537, opaque, slate background).
Output: spear_stone.png   RGBA, the stone spear cut out of the background (dust specks and the head halo removed)
        spear_holo.png    RGBA, a glowing point-cloud version of the same silhouette, synthesised from the stone image
        spear_layout.json sizes and the fraction-to-texture mapping the UI reads

Both layers have the same size, so one fraction maps to the same horizontal texture coordinate in both. The hologram is
synthesised (the 10% and 50% reference images show only parts of a hologram, from slightly different spear designs, so
no complete hologram exists in the references); its look is tuned against those two images by eye.

The cut-out thresholds are tuned for pelias-spear-100.png and are not general. Needs: numpy, scipy, opencv-python, Pillow.
Run:  bake_spear_textures.py <pelias-spear-100.png> <output folder> [--width 1600]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CANVAS = np.array([0x26, 0x2A, 0x31], np.float32)  # MatteSlate, only used for the preview sheet
HOLO_MID = np.array([100, 216, 230], np.float32) / 255.0  # hue ~185 deg: measured in the 10% reference (hologram dots)
HOLO_CORE = np.array([207, 254, 255], np.float32) / 255.0  # measured: brightest 0.1% of the hologram pixels
PAD_TEX = 20  # texture pixels of empty border around the spear, for glow
SYNTH_GAIN = 0.75
SYNTH_SRC_X = (400, 850)  # source-image columns where the reference glow is replaced by the synthesised one


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def background_model(a):
    h = a.shape[0]
    top = np.median(a[100:450].astype(np.float32), axis=0)
    bot = np.median(a[1000:1250].astype(np.float32), axis=0)
    t = np.clip((np.arange(h)[:, None, None] - 300) / 800.0, 0, 1)
    return top[None] * (1 - t) + bot[None] * t


def stone_mask(a):
    """Boolean mask of the stone spear in the full-size image."""
    h, w = a.shape[:2]
    af = a.astype(np.float32)
    gray = af.mean(2)
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2, 0))
    std5 = np.sqrt(np.maximum(cv2.blur(gray * gray, (5, 5)) - cv2.blur(gray, (5, 5)) ** 2, 0))
    d = np.abs(af - background_model(a)).sum(2)
    r, b = af[..., 0], af[..., 2]
    warm = cv2.dilate((((r - b) > -2) & (d > 12)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # dust specks
    band = np.zeros((h, w), bool)
    band[560:960, 150:2650] = True

    # 1. body: colour difference or grain, dust removed, then a smoothed top/bottom envelope per column
    thr = np.where(np.arange(w)[None, :] > 1840, 70, 45)
    m = ((d > thr) | (std > 3.0)) & band & (~warm)
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(4))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, disk(7))
    m = ndi.binary_fill_holes(m8 > 0)
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, [i + 1 for i in range(n) if sizes[i] > 20000])
    xs = np.where(m.any(0))[0]
    x0, x1 = xs.min(), xs.max()
    top = np.zeros(w)
    bot = np.zeros(w)
    for x in range(x0, x1 + 1):
        col = np.where(m[:, x])[0]
        top[x], bot[x] = col.min(), col.max()
    topf = ndi.median_filter(top[x0:x1 + 1], size=15, mode="nearest")
    botf = ndi.median_filter(bot[x0:x1 + 1], size=15, mode="nearest")
    env = np.zeros_like(m)
    for i, x in enumerate(range(x0, x1 + 1)):
        env[int(topf[i]):int(botf[i]) + 1, x] = True

    # 2. head (x > 2010): grain-based, so the smooth glow around the head is left out
    hm = (std5 > 3.0) & (d > 40)
    hm8 = cv2.morphologyEx(hm.astype(np.uint8), cv2.MORPH_CLOSE, disk(10))
    hm8 = ndi.binary_fill_holes(hm8 > 0).astype(np.uint8)
    hm8 = cv2.morphologyEx(hm8, cv2.MORPH_OPEN, disk(5))
    head = np.where(np.arange(w)[None, :] > 2010, (hm8 > 0) & env, env)
    lab, n = ndi.label(head)
    sizes = ndi.sum(head, lab, range(1, n + 1))
    head = np.isin(lab, [i + 1 for i in range(n) if sizes[i] > 20000])

    # 3. neck (x 1925..2121): the top edge from the grainy core is reliable; the bottom follows from the spear's axis
    core0 = cv2.morphologyEx((((std5 > 3.0) & (d > 40) & band)).astype(np.uint8), cv2.MORPH_OPEN, disk(6))
    lab, n = ndi.label(core0)
    sizes = ndi.sum(core0, lab, range(1, n + 1))
    core = np.isin(lab, [i + 1 for i in range(n) if sizes[i] > 2500]).astype(np.uint8)
    rec = cv2.dilate(core, disk(10)) & (((std5 > 3.0) & (d > 40) & band).astype(np.uint8))
    rec = cv2.morphologyEx(rec, cv2.MORPH_CLOSE, disk(9))
    rec = ndi.binary_fill_holes(rec > 0).astype(np.uint8)
    rec = cv2.morphologyEx(rec, cv2.MORPH_OPEN, disk(9))
    lab, n = ndi.label(rec)
    sizes = ndi.sum(rec, lab, range(1, n + 1))
    core = np.isin(lab, [i + 1 for i in range(n) if sizes[i] > 8000])
    m = head.copy()
    for x in range(1925, 2122):
        c = np.where(core[:, x])[0]
        if not c.size:
            continue
        m[:c.min() - 3, x] = False
        if x < 2112:
            m[int(2 * 768 - c.min() + 6):, x] = False
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(3))
    lab, n = ndi.label(m8)
    sizes = ndi.sum(m8, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i in range(n) if sizes[i] > 20000])


def fill_outside_with_nearest(rgb, inside):
    """RGB outside the mask takes the nearest inside colour, so bilinear sampling never pulls in a dark or light fringe."""
    idx = ndi.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return rgb[idx[0], idx[1]]


def crop_and_scale(a, mask, width):
    ys, xs = np.where(mask)
    xl, xr, yt, yb = xs.min(), xs.max(), ys.min(), ys.max()
    scale = (width - 2 * PAD_TEX) / float(xr - xl + 1)
    pad_src = int(round(PAD_TEX / scale))
    box = (xl - pad_src, yt - pad_src, xr + 1 + pad_src, yb + 1 + pad_src)
    soft = cv2.GaussianBlur(cv2.erode(mask.astype(np.uint8), disk(2)).astype(np.float32), (0, 0), 1.6)
    rgb = fill_outside_with_nearest(a.astype(np.float32), cv2.erode(mask.astype(np.uint8), disk(3)) > 0)
    rgb = rgb[box[1]:box[3], box[0]:box[2]]
    soft = soft[box[1]:box[3], box[0]:box[2]]
    w = width
    h = int(round((box[3] - box[1]) * (w / float(box[2] - box[0]))))
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(soft, (w, h), interpolation=cv2.INTER_AREA)
    layout = {"tex_w": w, "tex_h": h, "pad_px": PAD_TEX, "crop_box": [int(v) for v in box],
              "synth_from_tex": int(round((SYNTH_SRC_X[0] - box[0]) * (w / float(box[2] - box[0])))),
              "synth_to_tex": int(round((SYNTH_SRC_X[1] - box[0]) * (w / float(box[2] - box[0])))),
              "u0": PAD_TEX / float(w), "u1": 1.0 - PAD_TEX / float(w),
              "source_bbox": [int(xl), int(yt), int(xr), int(yb)], "scale": scale}
    return rgb, alpha, layout


def holo_from_reference(ref_rgb, mask, layout, shift_y=-3, boost=1.3):
    """The glowing layer cut from the author's 0% image (a complete hologram of the same spear, 1-3 px off the 100% image).
    Floating voxel squares are removed; the over-bright pommel (the seam glow at 0%) is toned down, because the seam glow
    is drawn live. Light-on-dark is converted to colour plus alpha so ImGui's ordinary alpha blend reproduces it."""
    h0, w0 = ref_rgb.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    a = cv2.warpAffine(ref_rgb, M, (w0, h0), borderMode=cv2.BORDER_REPLICATE)
    bg = background_model(a)
    diff = a.astype(np.float32) - bg
    # voxel squares in the cloud around the seam (src x 380..860): opening with a 7 px square keeps only blobs >= 7 px
    bright = (diff.max(2) > 40).astype(np.uint8)
    sq = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    sq[:, :380] = 0
    sq[:, 860:] = 0
    sq = cv2.dilate(sq, np.ones((5, 5), np.uint8))
    a = cv2.inpaint(a, sq, 6, cv2.INPAINT_TELEA)
    diff = a.astype(np.float32) - bg
    avail = np.clip(255.0 - bg, 1.0, None)
    alpha = np.clip((diff / avail).max(2), 0, 1)
    # the inpainted patches are smooth; sprinkle the same fine dots the rest of the hologram has (about 1 px in 5)
    rng = np.random.default_rng(11)
    dots = (rng.uniform(0, 1, alpha.shape) < 0.20) & (sq > 0)
    alpha = np.where(dots, np.maximum(alpha, rng.uniform(0.35, 0.75, alpha.shape)), alpha)
    safe = np.maximum(alpha, 1e-3)[..., None]
    col = np.clip((a.astype(np.float32) - bg * (1 - alpha[..., None])) / safe, 0, 255)
    col[alpha < 0.02] = np.array([150, 250, 255], np.float32)
    # keep what lies near the spear's silhouette; floating squares farther out are dropped
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    keep = np.exp(-(dist / 20.0) ** 2)
    alpha = alpha * keep
    # tone down the seam-glow pommel: gain 0.55 at src x<=420, rising to 1.0 at x>=760
    xs = np.arange(w0)[None, :]
    gain = np.clip(0.55 + 0.45 * (xs - 420) / 340.0, 0.55, 1.0)
    alpha = np.clip(alpha * gain * boost, 0, 1)
    x0, y0, x1, y1 = layout["crop_box"]
    w, h = layout["tex_w"], layout["tex_h"]
    pre = (col * alpha[..., None])[y0:y1, x0:x1]
    al = alpha[y0:y1, x0:x1]
    pre = cv2.resize(pre, (w, h), interpolation=cv2.INTER_AREA)
    al = cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA)
    col = np.where(al[..., None] > 1e-4, pre / np.maximum(al[..., None], 1e-4), np.array([150, 250, 255], np.float32))
    return np.clip(col / 255.0, 0, 1), np.clip(al, 0, 1)


def clean_silhouette(rgb, alpha, layout):
    """Stylised outline for the hologram: the shaft is mirrored about the spear's axis from its (reliable) top edge, with
    dust bumps removed; the leaf-shaped head keeps the real cut-out. Works in texture pixels."""
    h, w = alpha.shape
    sc = layout["scale"]
    src = (alpha > 0.5)
    top = np.full(w, np.nan)
    bot = np.full(w, np.nan)
    for x in range(w):
        c = np.where(src[:, x])[0]
        if c.size:
            top[x], bot[x] = c.min(), c.max()
    # axis in texture rows: source axis row 768 -> texture
    ymin_src = layout["source_bbox"][1] - int(round(PAD_TEX / sc))
    axis = (768 - ymin_src) * sc
    valid = ~np.isnan(top)
    t = np.where(valid, top, axis)
    # remove narrow upward bumps: grey opening of the "height" above the axis (axis - top) with a 14 px window
    height = axis - t
    base = ndi.median_filter(height, size=61, mode="nearest")           # the plain tube
    excess = np.clip(height - base, 0, None)
    Sg = (rgb @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
    rim = np.zeros(w)
    for x in range(w):
        if valid[x]:
            y0r = int(top[x]) + 3
            rim[x] = Sg[y0r:y0r + 6, x].mean()
    rim = ndi.uniform_filter1d(rim, 9)
    excess = np.where((excess > 6.0) & (rim > 0.36), excess, 0.0)       # collars: wider than 6 px and bright stone; dust is neither
    excess = ndi.grey_opening(excess, size=14)                           # narrower bumps are dust, not collars
    height = cv2.GaussianBlur((base + excess).astype(np.float32).reshape(1, -1), (0, 0), 1.8).reshape(-1)
    head_x0 = (1925 - layout["source_bbox"][0]) * sc + PAD_TEX
    sil = np.zeros((h, w), bool)
    for x in range(w):
        if not valid[x]:
            continue
        if x < head_x0:
            hh = height[x]
            sil[int(round(axis - hh)):int(round(axis + hh)) + 1, x] = True
        else:
            sil[:, x] = src[:, x]
    return sil


def hologram(rgb, alpha, layout, seed=7, lattice_dots=True, ring_gain=1.0, dens_scale=1.0):
    h, w = alpha.shape
    S = (rgb @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
    # smoother silhouette than the stone cut-out: the hologram outline is a clean line in the references
    smooth = cv2.GaussianBlur(clean_silhouette(rgb, alpha, layout).astype(np.float32), (0, 0), 1.4)
    inside = smooth > 0.5
    dist_in = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3)
    ring = np.where(inside, np.exp(-((dist_in - 1.0) / 1.1) ** 2), np.exp(-((dist_out - 0.6) / 0.8) ** 2)).astype(np.float32)
    inner_rim = np.exp(-((dist_in - 4.5) / 2.6) ** 2) * inside  # the faint second band just inside the outline

    Sb = cv2.GaussianBlur(S, (0, 0), 1.0)
    gx = cv2.Sobel(Sb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Sb, cv2.CV_32F, 0, 1, ksize=3)
    E = np.hypot(gx, gy)
    E = np.clip(E / (np.percentile(E[inside], 97) + 1e-6), 0, 1) ** 1.35
    dark = np.clip(cv2.GaussianBlur(S, (0, 0), 3.0) - Sb, 0, None)
    dark = np.clip(dark / (np.percentile(dark[inside], 97) + 1e-6), 0, 1) ** 1.2
    bright = np.clip((cv2.GaussianBlur(S, (0, 0), 2.0) - 0.36) / 0.38, 0, 1)  # the icy head is denser, as in the references
    F = np.clip(0.85 * E + 0.70 * dark, 0, 1)

    rng = np.random.default_rng(seed)
    pitch = 2.0
    ys_, xs_ = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing="ij")
    lattice = ((ys_ % 2 == 0) & (xs_ % 2 == 0)) if lattice_dots else (rng.uniform(0, 1, (h, w)) < 0.30)
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
    thr = bayer[(ys_ // 2) % 4, (xs_ // 2) % 4] * 0.85 + rng.uniform(0, 0.15, (h, w)).astype(np.float32)
    dens = (inside * (0.02 + 1.05 * F + 0.30 * bright * (0.35 + F) + 0.55 * inner_rim) + 1.6 * ring * ring_gain) * dens_scale
    D = (lattice & (dens > thr)).astype(np.float32) * rng.uniform(0.65, 1.0, (h, w)).astype(np.float32)
    D = np.clip(cv2.GaussianBlur(D, (0, 0), 0.7) * 3.2, 0, 1)

    a_fill = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 1.0) * (0.03 + 0.06 * np.clip(S * 1.6, 0, 1) + 0.12 * bright + 0.10 * inner_rim)
    a_ring = np.clip(cv2.GaussianBlur(ring, (0, 0), 0.5) * 0.9 * ring_gain, 0, 1)
    a_dot = D * 0.95
    glow_src = a_ring * 1.0 + a_dot * 0.55
    a_glow = np.clip(cv2.GaussianBlur(glow_src, (0, 0), 3.0) * 1.0 + cv2.GaussianBlur(glow_src, (0, 0), 8.0) * 0.8, 0, 0.55) * 0.7
    A = 1 - (1 - a_fill) * (1 - a_ring) * (1 - a_dot) * (1 - a_glow)
    heat = np.clip(a_dot * 0.9 + a_ring * 0.6 + a_fill * 0.5, 0, 1)[..., None]
    col = HOLO_MID[None, None] * (1 - heat) + HOLO_CORE[None, None] * heat
    return np.clip(col, 0, 1), np.clip(A, 0, 1)


def save_rgba(path, rgb01, alpha01):
    out = np.dstack([np.clip(rgb01 * 255 + 0.5, 0, 255), np.clip(alpha01 * 255 + 0.5, 0, 255)]).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(path, optimize=True)


def preview(stone_rgb, stone_a, holo_rgb, holo_a, frac, layout, out_path):
    h, w = stone_a.shape
    u = layout["u0"] + frac * (layout["u1"] - layout["u0"])
    cut = int(round(u * w))
    canvas = np.tile(CANVAS[None, None] / 255.0, (h, w, 1))
    s = stone_rgb / 255.0 * stone_a[..., None] + canvas * (1 - stone_a[..., None])
    hl = holo_rgb * holo_a[..., None] + canvas * (1 - holo_a[..., None])
    img = np.where((np.arange(w)[None, :, None] < cut), s, hl)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("outdir")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--holo-ref", help="pelias-spear-0.png: a complete hologram of the same spear; if absent the glow is synthesised")
    ap.add_argument("--preview", action="store_true", help="also write preview_<percent>.png")
    args = ap.parse_args()
    a = np.array(Image.open(args.source).convert("RGB"))
    mask = stone_mask(a)
    rgb, alpha, layout = crop_and_scale(a, mask, args.width)
    if args.holo_ref:
        holo_rgb, holo_a = holo_from_reference(np.array(Image.open(args.holo_ref).convert("RGB")), mask, layout)
        # the reference's first section (between the pommel and the first collar) is hidden under the seam's voxel cloud
        # and cannot be cleaned; there the synthesised glow (built from the stone's own cracks) is cross-faded in
        syn_rgb, syn_a = hologram(rgb, alpha, layout, lattice_dots=False, ring_gain=0.3, dens_scale=0.7)
        w_tex = layout["tex_w"]
        xs = np.arange(w_tex, dtype=np.float32)
        x_a, x_b = layout["synth_from_tex"], layout["synth_to_tex"]
        wgt = np.clip(np.minimum((xs - (x_a - 24)) / 24.0, ((x_b + 24) - xs) / 24.0), 0, 1)[None, :]
        pre = holo_rgb * holo_a[..., None] * (1 - wgt[..., None]) + syn_rgb * (syn_a * SYNTH_GAIN)[..., None] * wgt[..., None]
        al = holo_a * (1 - wgt) + syn_a * SYNTH_GAIN * wgt
        holo_rgb = np.where(al[..., None] > 1e-4, pre / np.maximum(al[..., None], 1e-4), holo_rgb)
        holo_a = np.clip(al, 0, 1)
        layout["glow_source"] = "reference image (first section synthesised)"
    else:
        holo_rgb, holo_a = hologram(rgb, alpha, layout)
        layout["glow_source"] = "synthesised"
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_rgba(out / "spear_stone.png", rgb / 255.0, alpha)
    save_rgba(out / "spear_holo.png", holo_rgb, holo_a)
    (out / "spear_layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    if args.preview:
        for frac in (0.1, 0.5, 1.0):
            preview(rgb, alpha, holo_rgb, holo_a, frac, layout, out / ("preview_%03d.png" % int(frac * 100)))
    print("wrote %s: texture %dx%d, u0=%.4f u1=%.4f" % (out, layout["tex_w"], layout["tex_h"], layout["u0"], layout["u1"]))


if __name__ == "__main__":
    sys.exit(main())
