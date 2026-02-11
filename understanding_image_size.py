import json
import numpy as np
import os
from pathlib import Path
DATASET_PATH = "/media/aiviz05/New Volume/Data/BISSMART/Detectron_dataset_yolo_coco_format/"

def analyze_coco_dims(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    if not images:
        print("No images found in JSON.")
        return

    # In COCO, 'width' and 'height' are standard keys
    # We want to find the 'short edge' and 'long edge' for each image
    short_edges = []
    long_edges = []

    for img in images:
        w, h = img['width'], img['height']
        short_edges.append(min(w, h))
        long_edges.append(max(w, h))

    stats = {
        "Short Edge (Min Size)": short_edges,
        "Long Edge (Max Size)": long_edges
    }

    print(f"--- Analysis for {json_path} ---")
    print(f"{'Metric':<20} | {'Short Edge':<12} | {'Long Edge':<12}")
    print("-" * 50)

    for metric in ["Average", "Median", "Min", "Max"]:
        if metric == "Average":
            s_val, l_val = np.mean(short_edges), np.mean(long_edges)
        elif metric == "Median":
            s_val, l_val = np.median(short_edges), np.median(long_edges)
        elif metric == "Min":
            s_val, l_val = np.min(short_edges), np.min(long_edges)
        elif metric == "Max":
            s_val, l_val = np.max(short_edges), np.max(long_edges)

        print(f"{metric:<20} | {s_val:<12.2f} | {l_val:<12.2f}")


# Usage
analyze_coco_dims(os.path.join(DATASET_PATH,'train.json'))