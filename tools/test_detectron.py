import os
from datetime import datetime
import torch
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from utils_predictions import (evaluate_predictions,
                               calculate_comprehensive_iou,
                               calculate_f1_metrics,
                               plot_and_save_matches)
from detectron2.structures import ImageList
from utils_load_data import CocoDataset,prepare_gn_annotations
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from utils_testing import SparseInst,device
from detectron2.checkpoint import DetectionCheckpointer

from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config

from utils_cv import resize_pad,preprocess_image , filter_prediction_exclusion_boxes , rescale_mask_to_original
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def run_detectron(image, model, device):
    # 1. ImageNet Constants scaled to 0-255 range (RGB order)
    # Mean: [0.485*255, 0.456*255, 0.406*255]
    # Std:  [0.229*255, 0.224*255, 0.225*255]
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(device).view(3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(device).view(3, 1, 1)

    # 2. Resize and Convert BGR -> RGB
    resized_image, _ = preprocess_image(image.data, (640, 640))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 3. Convert to Tensor (C, H, W) and move to device
    # Keeping values in 0-255 range as requested
    input_tensor = torch.as_tensor(rgb_image.transpose(2, 0, 1)).to(device).float()

    # 4. Normalize (Broadcasting across H and W)
    normalized_image = (input_tensor - pixel_mean) / pixel_std

    # 5. Wrap in ImageList for Batching & Padding (Stride 32)
    # This ensures the final tensor is (1, 3, H_padded, W_padded)
    image_list = ImageList.from_tensors([normalized_image], 32)

    # 6. Inference
    with torch.no_grad():
        results = model(image_list.tensor)

    return results, image_list.image_sizes[0]


def main(test_json,test_images,onnx_path,save_path,score_threshold,mask_threshold):
    dataset = CocoDataset(
        images_dir=test_images,
        annotation_file=test_json
    )
    args = default_argument_parser()
    args.add_argument("--fp16",
                      action="store_true",
                      help="support fp16 for inference")
    args = args.parse_args()
    cfg = setup(args)
    model = SparseInst(cfg)
    model.eval()
    model.to('cuda')
    print(model)
    DetectionCheckpointer(model,
                          save_dir=cfg.OUTPUT_DIR).resume_or_load(
                                    cfg.MODEL.WEIGHTS,
                                    resume=False)
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False

    model.to(device)
    model.eval()
    for image in dataset.images:
        image.load_data()
        original_shape = image.data.shape[:2]
        exclusion_boxes_modified = [[0,
                                     0,
                                     int(image.width * 1 / 10),
                                     int(image.height)],
                                    [int(image.width * 9 / 10), 0,
                                     int(image.width * 1 / 10),
                                     int(image.height)]]
        print(f"Processing: {image.filename} ({image.width}x{image.height})")
        exclusion_zone_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for box in exclusion_boxes_modified:
            cv2.rectangle(exclusion_zone_mask, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), 255, -1)
        mask_gt,gt_mask_cache = prepare_gn_annotations(image, exclusion_boxes_modified)

if __name__ == '__main__':
    test_path_json = "/media/aiviz05/New Volume/Data/TISSMART/Detectron/cvat-benchmark/cvat-output-coco.json"
    test_path_images = "/media/aiviz05/New Volume/Data/TISSMART/Detectron/cvat-benchmark/images"
    folder_containing_onnx_json = "/home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base_20260211_175840/final_model.onnx"
    model_score_threshold = 0.5
    model_mask_threshold = 0.5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'{timestamp}_onnxtest_runtime_results'
    checkpoints_dir = os.path.join('bin', new_run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    main(test_path_json,
         test_path_images,
         onnx_path = folder_containing_onnx_json,
         score_threshold=model_score_threshold,
         mask_threshold=model_mask_threshold,
         save_path=checkpoints_dir)