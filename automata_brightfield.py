import os
import glob
import csv
import cv2
import numpy as np
from scipy import ndimage as ndi


SRC_DIR = r"C:\Users\Lenovo ThinkPad P15\Desktop\EUM\Dipterv\KFKI\Mic\2024.11.11.kepek\BF"
DST_DIR = r"C:\Users\Lenovo ThinkPad P15\Desktop\EUM\Dipterv\KFKI\Mic\2024.11.11.kepek\Results BF"

# Brightfield defaultok
BG_KERNEL = 101           # háttérszűréshez Gauss blur sigma ~ kernel/3
METHOD = "adaptive"
ADAPTIVE_BLOCK = 71
ADAPTIVE_OFFSET = -5
OPEN_K = 4
CLOSE_K = 5
MIN_AREA = 1400           
MAX_AREA = 20000          
ALPHA = 0.60
DOWNSAMPLE = 2


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Nem sikerült beolvasni: {path}")
    return img

def normalize_uint8(im):
    im = im.astype(np.float32)
    im -= im.min()
    rng = im.max() if im.max() > 0 else 1.0
    return (im * (255.0 / rng)).astype(np.uint8)

def background_correct(im_u8, kernel):
    sigma = max(1, int(kernel/3))
    bg = cv2.GaussianBlur(im_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.subtract(im_u8, bg)

def threshold_image(inv_u8, method, block, offset):
    if method == "adaptive":
        blk = block if block % 2 == 1 else (block + 1)
        th = cv2.adaptiveThreshold(inv_u8, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   blk, offset)
    else:
        _, th = cv2.threshold(inv_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def postprocess(bin_img, open_k, close_k):
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k)
    return bin_img

def fill_borders_and_holes(bin_img):
    h, w = bin_img.shape[:2]
    im_pad = cv2.copyMakeBorder(bin_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((h+2+2, w+2+2), np.uint8)
    cv2.floodFill(im_pad, mask, (0, 0), 255)
    cv2.floodFill(im_pad, mask, (w+1, 0), 255)
    cv2.floodFill(im_pad, mask, (0, h+1), 255)
    cv2.floodFill(im_pad, mask, (w+1, h+1), 255)
    filled_inv = cv2.bitwise_not(im_pad[1:-1, 1:-1])
    out = cv2.bitwise_or(bin_img, filled_inv)
    bin_bool = out > 0
    bin_filled = ndi.binary_fill_holes(bin_bool)
    return (bin_filled.astype(np.uint8)) * 255

def filter_and_relabel(labels_im, min_area, max_area):
    if labels_im.max() == 0:
        return labels_im
    mask = (labels_im > 0).astype(np.uint8) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(labels_im, dtype=bool)
    for lbl in range(1, num):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            keep |= (labels_im > 0) & (lab == lbl)
    filtered = np.where(keep, labels_im, 0).astype(np.int32)
    u = np.unique(filtered); u = u[u > 0]
    out = np.zeros_like(filtered, dtype=np.int32)
    for new, old in enumerate(u, start=1):
        out[filtered == old] = new
    return out

def colored_overlay(gray_u8, labels_im, alpha):
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    if labels_im.max() == 0:
        return base
    m = (labels_im.astype(np.float32) / labels_im.max() * 255).astype(np.uint8)
    color_mask = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    cells = (labels_im > 0).astype(np.uint8)
    cells3 = np.repeat(cells[..., None], 3, axis=2)
    blend = (base * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return np.where(cells3 == 1, blend, base)

def process_image(img_gray):
    img_u8 = normalize_uint8(img_gray)

    s = max(1, int(DOWNSAMPLE))
    work = cv2.resize(img_u8, None, fx=1.0/s, fy=1.0/s, interpolation=cv2.INTER_AREA) if s > 1 else img_u8.copy()
    min_area = max(1, int(MIN_AREA / (s*s)))
    max_area = max(1, int(MAX_AREA / (s*s)))

    corrected = background_correct(work, BG_KERNEL)
    inv = 255 - normalize_uint8(corrected)

    bin_img = threshold_image(inv, METHOD, ADAPTIVE_BLOCK, ADAPTIVE_OFFSET)

    tw = 512
    th = max(1, int(bin_img.shape[0] * (tw / max(1, bin_img.shape[1]))))
    small = cv2.resize(bin_img, (tw, th), interpolation=cv2.INTER_NEAREST)
    if (small > 0).mean() > 0.70 and METHOD == "adaptive":
        blk = ADAPTIVE_BLOCK if (ADAPTIVE_BLOCK % 2 == 1) else (ADAPTIVE_BLOCK + 1)
        adj_offset = min(0, ADAPTIVE_OFFSET + 4)
        bin_img = cv2.adaptiveThreshold(inv, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        blk, adj_offset)
        small2 = cv2.resize(bin_img, (tw, th), interpolation=cv2.INTER_NEAREST)
        if (small2 > 0).mean() > 0.70:
            _, bin_img = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bin_img = postprocess(bin_img, OPEN_K, CLOSE_K)
    bin_img = fill_borders_and_holes(bin_img)
    _, labels_cc = cv2.connectedComponents(bin_img, connectivity=8)
    labels_cc = labels_cc.astype(np.int32)

    labels_clean_small = filter_and_relabel(labels_cc, min_area, max_area)

    # vissza full felbontásra
    if s > 1:
        labels_up = cv2.resize(labels_clean_small, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        labels_up = labels_clean_small

    overlay = colored_overlay(img_u8, labels_up, ALPHA)
    return labels_up, overlay

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_outputs(base_name, labels, overlay, out_dir):
    ov_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(ov_path, overlay)

    cell_ids = np.unique(labels)
    cell_ids = cell_ids[cell_ids > 0]
    count = int(cell_ids.size)

    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Total Cell Count", count])
        w.writerow(["Cell ID", "Area (px)"])
        for cid in cell_ids:
            area = int(np.count_nonzero(labels == cid))
            w.writerow([cid, area])

def main():
    ensure_dir(DST_DIR)
    paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.png")))
    if not paths:
        print("Nem találtam PNG fájlokat a forrás mappában.")
        return

    for p in paths:
        try:
            img = imread_unicode(p)
            labels, overlay = process_image(img)
            base = os.path.splitext(os.path.basename(p))[0]
            save_outputs(base, labels, overlay, DST_DIR)
            print(f"Kész: {base}  (sejtszám={int(labels.max())})")
        except Exception as e:
            print(f"Hiba: {p}\n  {e}")

if __name__ == "__main__":
    main()
