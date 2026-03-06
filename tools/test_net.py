import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import json
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, print_csv_format

sys.path.append(".")
from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config
from sparseinst import COCOMaskEvaluator
from detectron2.data.datasets import register_coco_instances
from datetime import datetime
from pycocotools import mask as mask_util


device = torch.device('cuda:0')
dtype = torch.float32

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from utils_testing import SparseInst,synchronize,process_batched_inputs
pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)





def test_sparseinst_speed(cfg, fp16=False):
    device = torch.device('cuda:0')

    model = SparseInst(cfg)
    model.eval()
    model.to(device)
    print(model)
    size = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(cfg.OUTPUT_DIR, f"inference_{timestamp}")

    evaluator = COCOMaskEvaluator(
        cfg.DATASETS.TEST[0], ("segm",), False, output_folder)

    evaluator.reset()
    model.to(device)
    model.eval()

    # Get metadata for the dataset (classes, colors, etc.)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    durations = []
    current_visualize_name = os.path.join(output_folder,f"visualization_output")
    # Create a directory for outputs if it doesn't exist
    os.makedirs(current_visualize_name, exist_ok=True)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    durations = []
    json_results = []

    with torch.amp.autocast('cuda', enabled=False):
        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                images, resized_size, ori_size = process_batched_inputs(inputs)
                synchronize()
                start_time = time.perf_counter()
                output = model(images, resized_size, ori_size)
                synchronize()
                end = time.perf_counter() - start_time
                instances = output.to("cpu")
                predicted_masks = instances.pred_masks
                predicted_scores = instances.scores
                for i, (mask,score) in enumerate(zip(predicted_masks, predicted_scores)):
                    img_path = os.path.basename(inputs[0]["file_name"])
                    mask_np = mask.numpy()
                    score_np = score.numpy()
                    rle = mask_util.encode(np.asfortranarray(mask_np.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")  # make JSON serializable
                    json_results.append({
                        "image_id": os.path.basename(img_path),
                        "score": float(score_np),
                        "segmentation": rle
                    })


                durations.append(end)
                if idx % 1000 == 0:
                    print("process: [{}/{}] fps: {:.3f}".format(idx,
                                                                len(data_loader), 1/np.mean(durations[100:])))
                evaluator.process(inputs, [{"instances": output}])

                if idx < 100:
                    # 1. Get the image as it exists in the input (usually resized/transformed)
                    img_tensor = inputs[0]["image"].permute(1, 2, 0).cpu().numpy()

                    # 2. IMPORTANT: SparseInst resizes masks to 'ori_size'.
                    # We must ensure the visualizer's canvas matches the mask shape.
                    # We use 'ori_size' which is (height, width)
                    h_ori, w_ori = ori_size

                    # If the image in 'inputs' is already resized, we should resize it
                    # to match the masks before visualizing
                    img_for_vis = cv2.resize(img_tensor, (w_ori, h_ori))

                    # 3. Initialize Visualizer with the correctly sized image
                    visualizer = Visualizer(img_for_vis, metadata=metadata, scale=1.0)

                    if output is not None:
                        # Move output to CPU
                        cpu_output = output.to("cpu")

                        # 4. Draw
                        vis_output = visualizer.draw_instance_predictions(cpu_output)
                        vis_img = vis_output.get_image()

                        # 5. Save (BGR for cv2)
                        out_img_name = f'res{idx}.jpg'
                        out_filename = os.path.join(current_visualize_name, out_img_name)
                        cv2.imwrite(out_filename, vis_img[:, :, ::-1])
                        print(f"Saved: {out_filename} with shape {vis_img.shape}")
                   # evaluate
    results = evaluator.evaluate()
    print_csv_format(results)
    # Save prediction results
    json_out_path = os.path.join(output_folder, "detectron_predictions.json")
    with open(json_out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    latency = np.mean(durations[100:])
    fps = 1 / latency
    print("speed: {:.4f}s FPS: {:.2f}".format(latency, fps))


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


if __name__ == '__main__':

    args = default_argument_parser()
    args.add_argument("--fp16", action="store_true",
                      help="support fp16 for inference")
    args = args.parse_args()
    print("Command Line Args:", args)
    # register_coco_instances(
    #     "pills_train", {}, f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/train.json", f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/train/imgs"
    # )
    #
    # register_coco_instances(
    #     "pills_val", {}, f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/val.json", f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/val/imgs"
    # )


    # register_coco_instances(
    #     "pills_train", {},
    #     f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/train.json",
    #     f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/train/imgs"
    # )

    register_coco_instances(
        "pills_val",
        {},
        f"/home/mehran/Desktop/SparseInst_DATA/cvat-benchmark/cvat-output-coco.json",
        f"/home/mehran/Desktop/SparseInst_DATA/cvat-benchmark/images"
    )
    cfg = setup(args)
    test_sparseinst_speed(cfg, fp16=args.fp16)
