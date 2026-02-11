import json
import pandas as pd
from typing import Dict, List
from loguru import logger
from pathlib import Path
from pycocotools import mask as mask_util
import numpy as np
import cv2
import os
def get_dataset_split(filename: str) -> str:
    """
    Infers the dataset split/type from the filename.
    Assumes filename format: 'type_description-0000.ext'
    """
    if "-" in filename:
        return filename.split("-")[0]
    return "other"
def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) between two binaray masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union
def is_rectangle_overlapping(bbox, exclusion_zone):
    x1, y1, w1, h1 = bbox
    x2, y2, w2, h2 = exclusion_zone
    if (x1 + w1 < x2) or (x2 + w2 < x1) or (y1 + h1 < y2) or (y2 + h2 < y1):
        return False
    else:
        return True
def polygon_to_mask(polygon: List[List[int]], width: int, height: int) -> np.ndarray:
    """
    Converts a list of polygon points into a binary mask image.

    Args:
        polygon: A list of [x, y] points defining the contour.
        width: The width of the image/mask.
        height: The height of the image/mask.

    Returns:
        A binary numpy array (mask) of shape (height, width) with values 0 or 1.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if not polygon:
        return mask

    # cv2.fillPoly expects a list of contours, where each contour is a NumPy array.
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Fill the polygon area in the mask with 1
    cv2.fillPoly(mask, [pts], 1)

    return mask


import json
from pathlib import Path


def load_coco_gt_map(coco_json_path: Path):
    with open(coco_json_path, "r") as f:
        data = json.load(f)

    # 1. Map category IDs to names
    cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # 2. Build the image map
    # Key: file_name, Value: metadata + empty annotations list
    gt_map = {}
    for img in data['images']:
        gt_map[img['file_name']] = {
            "width": img['width'],
            "height": img['height'],
            "annotations": {}  # We'll store annotations as a dict {ann_id: data}
        }

    # 3. Fill annotations into the correct image
    # Note: coco_json_path may use integer image_ids,
    # but your function uses filenames (image_id in pred["image_id"]).
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    for ann in data['annotations']:
        filename = img_id_to_filename.get(ann['image_id'])
        if filename and filename in gt_map:
            ann_id = ann['id']
            gt_map[filename]["annotations"][ann_id] = {
                "polygons": ann['segmentation'],
                "class_name": cat_id_to_name.get(ann['category_id'], "unknown")
            }

    return gt_map
def calculate_detectron_iou(
        gt_map: Dict,
        detectron_json_path: Path,
        model_source: str = "Detectron2",
        output_path: str = "output",
        ignore_border: int = 50  # pixels to ignore on left/right
) -> pd.DataFrame:
    """
    Calculates IoU for Detectron2 predictions (COCO RLE) against Ground Truth masks,
    ignoring left/right borders up to `ignore_border` pixels.
    """
    logger.info(f"--- Calculating Per-Instance IoU for {model_source} (COCO RLE format) ---")

    if not detectron_json_path.exists():
        logger.error(f"Detectron2 JSON not found at {detectron_json_path}")
        return pd.DataFrame()

    with open(detectron_json_path, "r") as f:
        detectron_data = json.load(f)

    instance_results = []

    for pred in detectron_data:
        image_id = pred["image_id"]
        if image_id not in gt_map:
            logger.warning(f"Skipping {image_id}: not found in Ground Truth.")
            continue

        gt_file_data = gt_map[image_id]
        width, height = gt_file_data["width"], gt_file_data["height"]
        dataset_split = get_dataset_split(image_id)
        gt_anns = gt_file_data["annotations"]

        # Decode Detectron2 RLE mask → binary
        rle = pred["segmentation"]
        pred_mask = mask_util.decode(rle)
        pred_mask = (pred_mask > 0).astype(np.uint8)

        # --- Ignore left/right border ---
        pred_mask[:, :ignore_border] = 0        # left
        pred_mask[:, width-ignore_border:] = 0  # right

        iou_best = 0.0
        best_ann_id = None
        gt_area = 0
        pred_area = pred_mask.sum()
        area_ratio = 0.0
        class_name = "unknown"

        # Match this prediction with the GT instance that gives highest IoU
        for ann_id, gt_ann in gt_anns.items():
            gt_polygon = gt_ann["polygons"]
            if not gt_polygon:
                continue

            gt_mask = polygon_to_mask(gt_polygon, width, height)
            iou = calculate_iou(gt_mask, pred_mask)

            if iou > iou_best:
                iou_best = iou
                best_ann_id = ann_id
                gt_area = gt_mask.sum()
                class_name = gt_ann["class_name"]

        if gt_area > 0:
            area_ratio = pred_area / gt_area

        instance_results.append({
            "filename": image_id,
            "dataset_split": dataset_split,
            "annotation_id": best_ann_id if best_ann_id else "N/A",
            "class_name": class_name,
            "IoU": iou_best,
            "GT_Area": gt_area,
            "Pred_Area": pred_area,
            "Area_Ratio": area_ratio,
            "prediction_source": model_source
        })

    df_instance = pd.DataFrame(instance_results)
    df_instance.to_csv(os.path.join(output_path,"results.csv"))
    logger.info(f"Finished processing {len(df_instance)} Detectron2 instances.")
    return df_instance
def main():
    gn_path = Path("/media/aiviz05/New Volume/Data/TISSMART/Detectron/cvat-benchmark/cvat-output-coco.json")
    gt_map = load_coco_gt_map(gn_path)
    path_to_results = '/home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base_longrun/inference_20260211_171838'

    detectron_json_path = Path(os.path.join(path_to_results,"detectron_predictions.json"))
    df_detectron_instance = calculate_detectron_iou(gt_map,
                                                    detectron_json_path,
                                                    model_source="Detectron2",
                                                    output_path=path_to_results)

if __name__ == "__main__":
    main()
