# inspect /content/CIFAR-10-C .npy files

import os
import numpy as np
import json
from pathlib import Path
from PIL import Image
import math

ROOT = Path("/content/CIFAR-10-C")  
OUT = Path("results")
SAMPLES = OUT / "samples"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)

def is_image_array(arr: np.ndarray):
    """Heuristic check whether array looks like images (H/W/C shape and small H/W dimensions)."""
    if arr.ndim == 4:
        n, h, w, c = arr.shape
        if c in (1,3) and (h <= 1024 and w <= 1024):
            return True
    if arr.ndim == 4 and arr.shape[1] in (1,3) and arr.shape[2] <= 1024:
        # maybe N, C, H, W
        return True
    return False

def to_uint8_image(arr: np.ndarray):
    """Convert array slice to uint8 HxW(xC) for saving. Accepts floats in [0,1] or ints [0,255]."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        img = np.clip(arr, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)
    else:
        img = arr.astype(np.uint8)
    # if CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1,3):
        img = np.transpose(img, (1,2,0))
    # if grayscale single channel -> HxW
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:,:,0]
    return img

report = {}
files = sorted([p for p in ROOT.iterdir() if p.suffix == ".npy"])
if not files:
    raise SystemExit(f"No .npy files found in {ROOT}. Check path and rerun.")

print(f"Found {len(files)} .npy files in {ROOT}\n")
for p in files:
    name = p.name
    print(f"Loading {name} ...", end=" ", flush=True)
    arr = np.load(p)
    print("done.")
    info = {
        "path": str(p),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes) if hasattr(arr, "nbytes") else None,
    }

    # basic statistics if numeric
    try:
        info["min"] = float(np.min(arr)) if arr.size>0 else None
        info["max"] = float(np.max(arr)) if arr.size>0 else None
        info["mean"] = float(np.mean(arr)) if arr.size>0 else None
        info["std"] = float(np.std(arr)) if arr.size>0 else None
    except Exception as e:
        info["stats_error"] = str(e)

    # Labels detection: likely 1D vector with length 10000
    if arr.ndim == 1:
        unique, counts = np.unique(arr, return_counts=True)
        info["is_label_array"] = True
        info["unique_count"] = int(len(unique))
        info["length"] = int(arr.shape[0])
        info["unique_sample"] = unique[:10].tolist()
        info["counts_sample"] = counts[:10].tolist()
        # Save sample of labels to JSON-friendly list (first 200)
        info["first_labels_sample"] = arr[:200].tolist()
    else:
        info["is_label_array"] = False

    # Image file heuristics and per-severity analysis
    if is_image_array(arr):
        info["looks_like_images"] = True
        # Normalize shape to (N, H, W, C)
        if arr.ndim == 4:
            if arr.shape[1] in (1,3) and arr.shape[2] <= 1024:
                # shape (N, C, H, W) -> convert to (N, H, W, C)
                if arr.shape[1] in (1,3) and arr.shape[3] <= 4:
                    # unexpected ordering, but handle common CHW or HWC
                    pass
            # Try to detect ordering
            n = arr.shape[0]
            if arr.shape[-1] in (1,3) and arr.shape[1] not in (1,3):
                # likely (N,H,W,C)
                n, h, w, c = arr.shape
                arr_hwc = arr
            elif arr.shape[1] in (1,3):
                # likely (N,C,H,W) -> convert
                n, c, h, w = arr.shape
                arr_hwc = np.transpose(arr, (0,2,3,1))
            else:
                # fallback: treat as (N,H,W,C)
                n, h, w, c = arr.shape
                arr_hwc = arr
        else:
            info["image_format_note"] = "unexpected ndim for images"
            arr_hwc = arr

        info["num_images_total"] = int(arr_hwc.shape[0])

        # detect severity grouping: divisible by 10000 (common CIFAR-10-C)
        if arr_hwc.shape[0] % 10000 == 0:
            groups = arr_hwc.shape[0] // 10000
            info["num_severities_in_file"] = int(groups)
            info["per_severity_counts"] = [10000] * groups
            # compute simple per-severity stats (mean pixel value per channel)
            per_sev = []
            for s in range(groups):
                start = s * 10000
                end = (s + 1) * 10000
                subset = arr_hwc[start:end].astype(np.float32) / (255.0 if arr_hwc.dtype != np.float32 and arr_hwc.dtype != np.float64 else 1.0)
                # per-channel mean (HWC -> axis=(0,1,2), channel last)
                mean_channels = list(np.mean(subset, axis=(0,1,2)).tolist())
                std_channels = list(np.std(subset, axis=(0,1,2)).tolist())
                per_sev.append({"mean_channels": mean_channels, "std_channels": std_channels})
                # save one representative image per severity (first image)
                try:
                    img = to_uint8_image(subset[0])
                    fname = SAMPLES / f"{p.stem}_s{s+1}.png"
                    Image.fromarray(img).save(fname)
                    per_sev[-1]["saved_sample"] = str(fname)
                except Exception as e:
                    per_sev[-1]["saved_sample_error"] = str(e)
            info["per_severity_stats"] = per_sev
        else:
            # not divisible by 10000: report number and attempt to treat as single-severity set
            info["num_severities_in_file"] = None
            info["per_severity_counts"] = [int(arr_hwc.shape[0])]
            try:
                subset = arr_hwc.astype(np.float32) / (255.0 if arr_hwc.dtype != np.float32 and arr_hwc.dtype != np.float64 else 1.0)
                mean_channels = list(np.mean(subset, axis=(0,1,2)).tolist())
                std_channels = list(np.std(subset, axis=(0,1,2)).tolist())
                info["per_severity_stats"] = [{"mean_channels": mean_channels, "std_channels": std_channels}]
                # save sample
                img = to_uint8_image(subset[0])
                fname = SAMPLES / f"{p.stem}_s1.png"
                Image.fromarray(img).save(fname)
                info["per_severity_stats"][0]["saved_sample"] = str(fname)
            except Exception as e:
                info["per_severity_stats"] = [{"error": str(e)}]
    else:
        info["looks_like_images"] = False

    report[name] = info

# Write JSON report
report_path = OUT / "inspect_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("\nInspection complete.")
print(f"Report written to: {report_path}")
print(f"Sample images (one per severity) saved under: {SAMPLES}")
print("\nQuick summary (file : num_images / num_severities):")
for fname, info in report.items():
    if info.get("looks_like_images"):
        print(f"- {fname}: {info.get('num_images_total')} images, severities={info.get('num_severities_in_file')}")
    elif info.get("is_label_array"):
        print(f"- {fname}: labels length={info.get('length')}, unique={info.get('unique_count')}")
    else:
        print(f"- {fname}: shape={info.get('shape')} (non-image)")

# Print path to report
print(str(report_path))
