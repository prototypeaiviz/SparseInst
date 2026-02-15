import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from utils_predictions import (evaluate_predictions,
                               calculate_comprehensive_iou,
                               calculate_f1_metrics,
                               plot_and_save_matches)
from utils_load_data import CocoDataset,prepare_gn_annotations
from utils_cv import (preprocess_image ,
                      filter_prediction_exclusion_boxes ,
                      rescale_mask_to_original)
def run_onnx(ort_session,
             image):
    # resized_image, resize_scale = resize_pad(image.data, min_size_test=640, max_size_test=853, canvas_w=640,
    #                                          canvas_h=640)
    resized_image, resize_scale = preprocess_image(image.data,(640,640))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    rgb_image_float = rgb_image.astype('float32')
    rgb_image_float = rgb_image_float.transpose(2, 0, 1)
    rgb_image_float_expanded = np.expand_dims(rgb_image_float, 0)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(
        None,  # Returns all output nodes
        {input_name: rgb_image_float_expanded}
    )
    scores = outputs[0][0]
    masks = outputs[1][0]
    return (scores,
            masks,
            resize_scale)

def prepare_post_processing(image,
                            scores,
                            masks,
                            exclusion_boxes_modified,
                            score_threshold,
                            mask_threshold,
                            resize_scale,
                            original_shape):
    empty_mask = np.zeros_like(image.data, dtype=np.uint8)
    prediction_mask_cache = {}
    ann_id = 0
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        # 1. Get the mask and resize to original image dimensions
        prediction_mask = masks[i]
        resize_prediction_mask = rescale_mask_to_original(prediction_mask, original_shape, resize_scale)

        # If mask is float, convert to binary (0 or 255)
        resize_prediction_mask = (resize_prediction_mask > mask_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(resize_prediction_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=lambda cnt: cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3])
            x, y, w, h = cv2.boundingRect(largest_contour)
            largest_bbox = [x, y, w, h]
        else:
            continue
        if not filter_prediction_exclusion_boxes(image.height,
                                                 image.width,
                                                 largest_bbox,
                                                 exclusion_boxes_modified):
            continue
        empty_mask[resize_prediction_mask > 0] = 255
        ann_id +=1
        prediction_mask_cache[ann_id] = resize_prediction_mask
    return (empty_mask ,
            prediction_mask_cache)
def main(test_json,test_images,onnx_path,save_path,score_threshold,mask_threshold):
    dataset = CocoDataset(
        images_dir=test_images,
        annotation_file=test_json
    )
    providers = ['CUDAExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    results_over_all_images = []
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
        scores,masks, resize_scale= run_onnx(ort_session, image)
        empty_mask,prediction_mask_cache = prepare_post_processing(image,
                                            scores,
                                            masks,
                                            exclusion_boxes_modified,
                                            score_threshold,
                                            mask_threshold,
                                            resize_scale,
                                            original_shape)
        print(f'We currently have this number of prediction mask cache {len(prediction_mask_cache)}')
        print(f'We currently have this number of gt mask cache {len(gt_mask_cache)}')
        results = evaluate_predictions( prediction_mask_cache=prediction_mask_cache,
                                        gt_mask_cache=gt_mask_cache)
        num_gts = len(gt_mask_cache)
        if results['tp'] > 0:
            avg_iou = calculate_comprehensive_iou(results, num_gts)
            metric_dictionary = calculate_f1_metrics(results)
            metric_dictionary["avg_iou"] = avg_iou
        else:
            avg_iou= 0
            metric_dictionary = {
                                    "precision": 0,
                                    "recall": 0,
                                    "f1_score": 0,
                                    "avg_iou":avg_iou
                                }
        metric_dictionary["image_name"] = image.filename
        metric_dictionary["dataset_name"] = image.filename.split("-")[0]
        debug_file_name = f'Debug_{image.filename}'
        save_file_name = os.path.join(save_path,debug_file_name)
        plot_and_save_matches(image.data,results["matches"],save_file_name)
        results_over_all_images.append(metric_dictionary)
    current_dataframe = pd.DataFrame(results_over_all_images)
    current_dataframe.to_csv(os.path.join(save_path,"results.csv"),index=False)
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