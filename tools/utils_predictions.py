import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
def get_contour_from_mask(mask):
    """Helper to convert a binary mask to a simplified polygon."""
    if mask is None or np.sum(mask) == 0:
        return None
    # Ensure mask is uint8
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Return the largest contour if there are multiple blobs for one instance
    return max(contours, key=cv2.contourArea)

def evaluate_predictions(prediction_mask_cache, gt_mask_cache, iou_threshold=0.5):
    """
    Calculates optimal IoU matching between predictions and ground truths.
    """
    pred_ids = list(prediction_mask_cache.keys())
    gt_ids = list(gt_mask_cache.keys())

    num_preds = len(pred_ids)
    num_gts = len(gt_ids)

    if num_preds == 0 or num_gts == 0:
        return {"matches": [], "tp": 0, "fp": num_preds, "fn": num_gts}

    # 1. Initialize the IoU Matrix
    # Rows = Predictions, Cols = Ground Truths
    iou_matrix = np.zeros((num_preds, num_gts))

    # 2. Fill the matrix
    for i, p_id in enumerate(pred_ids):
        p_mask = prediction_mask_cache[p_id] > 0  # Ensure boolean
        for j, g_id in enumerate(gt_ids):
            g_mask = gt_mask_cache[g_id] > 0

            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()

            iou_matrix[i, j] = intersection / union if union > 0 else 0

    # 3. Solve the Assignment Problem (Maximizing total IoU)
    # linear_sum_assignment minimizes, so we negate the matrix
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

    matches = []
    matched_preds = set()
    matched_gts = set()

    # 4. Filter matches by threshold
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[p_idx, g_idx]
        if iou >= iou_threshold:
            p_id = pred_ids[p_idx]
            g_id = gt_ids[g_idx]

            # Convert masks to contours for storage
            p_poly = get_contour_from_mask(prediction_mask_cache[p_id])
            g_poly = get_contour_from_mask(gt_mask_cache[g_id])

            matches.append({
                "pred_id": p_id,
                "gt_id": g_id,
                "iou": iou,
                "pred_poly": p_poly,  # Saved as [N, 1, 2]
                "gt_poly": g_poly  # Saved as [N, 1, 2]
            })
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)

    # 5. Calculate Metrics
    tp = len(matches)  # True Positives
    fp = num_preds - len(matched_preds)  # False Positives (Predictions with no GT)
    fn = num_gts - len(matched_gts)  # False Negatives (GTs never found)

    return {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_matrix": iou_matrix
    }


def plot_and_save_matches(image, matches, save_path="debug_matches.jpg"):
    """
    Plots GT vs Predictions and saves the image.
    GT is Green, Prediction is Cyan.
    """
    # Create a copy so we don't modify the original image
    vis_img = image.copy()

    for match in matches:
        g_poly = match["gt_poly"]
        p_poly = match["pred_poly"]
        iou = match["iou"]

        # 1. Draw Ground Truth (Green)
        if g_poly is not None:
            cv2.drawContours(vis_img, [g_poly], -1, (0, 255, 0), 2)

        # 2. Draw Prediction (Cyan)
        if p_poly is not None:
            cv2.drawContours(vis_img, [p_poly], -1, (255, 255, 0), 2)

            # 3. Add IoU text label near the prediction
            # Get the top-left point of the prediction to place text
            x, y, w, h = cv2.boundingRect(p_poly)
            label = f"IoU: {iou:.2f}"
            cv2.putText(vis_img, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save to disk
    cv2.imwrite(save_path, vis_img)
    print(f"Visualization saved to {save_path}")
    return vis_img
def calculate_comprehensive_iou(results, num_gts):
    # Sum of IoUs from successful matches
    total_iou = sum(m['iou'] for m in results['matches'])

    # We divide by the TOTAL number of Ground Truths, not just the matches.
    # Any FN (False Negative) effectively adds a 0.0 to the numerator.
    if num_gts == 0:
        return 0.0

    mean_iou = total_iou / num_gts
    return mean_iou


def calculate_f1_metrics(results):
    tp = results['tp']
    fp = results['fp']
    fn = results['fn']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }