import cv2
import numpy as np
def rescale_mask_to_original(padded_mask, original_shape, scaling):
    """
    Reverses the resize_pad operation to map a mask back to original image space.

    :param padded_mask: The output from the model (the 853x853 mask).
    :param original_shape: Tuple of (orig_h, orig_w) of the raw input image.
    :param scaling: The scale factor returned by the resize_pad function.
    :return: Mask resized back to original_shape.
    """
    orig_h, orig_w = original_shape

    # 1. Calculate the dimensions of the image before it was padded
    # We use the same rounding logic (int + 0.5) used in the forward pass
    unpadded_h = int(orig_h * scaling + 0.5)
    unpadded_w = int(orig_w * scaling + 0.5)

    # 2. Un-pad: Crop the top-left corner where the image was pasted
    # This removes the bottom-right padding
    cropped_mask = padded_mask[0:unpadded_h, 0:unpadded_w]

    # 3. Un-resize: Scale back to the original dimensions
    # Using INTER_NEAREST for masks to preserve class labels (0, 1, 2...)
    # If the mask is a probability map (0.0 to 1.0), use INTER_LINEAR
    original_size_mask = cv2.resize(
        cropped_mask,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    return original_size_mask
def preprocess_image(image_array, input_shape):

    original_height, original_width = image_array.shape[:2]
    h, w = input_shape
    # Step 1: Aspect-ratio preserving resize
    scale = min(h / original_height, w / original_width)
    new_height = int(round(original_height * scale))
    new_width = int(round(original_width * scale))

    if max(new_height, new_width) > max(h, w):
        scale = max(h, w) / max(new_height, new_width)
        new_height = int(round(original_height * scale))
        new_width = int(round(original_width * scale))

    resized_image = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((h, w, 3), 128, dtype=np.uint8)
    padded_image[:new_height, :new_width] = resized_image

    return padded_image, scale
def resize_pad(image,
               min_size_test=640,
               max_size_test=853,
               canvas_w=853,
               canvas_h=853,
               pad_color=(124, 116, 104)):
    """
    OpenCV implementation of the resize and pad logic.
    :param image: NumPy array (BGR or RGB)
    :param min_size_test: Goal for the shortest edge
    :param max_size_test: Maximum allowed for the longest edge
    :param canvas_w: Final output dimensions (the 'pad' size)
    :param canvas_h: Final output dimensions (the 'pad' size)
    :param pad_color: Tuple of 3 values for padding
    """
    # 1. Get characteristics (OpenCV shape is [height, width, channels])
    height, width = image.shape[:2]

    # 2. Calculate initial scale based on shortest edge
    size = float(min_size_test)
    pre_scale = size / min(height, width)

    if height < width:
        newh, neww = size, pre_scale * width
    else:
        newh, neww = pre_scale * height, size

    # 3. Constrain by max_size_test (longest edge)
    if max(newh, neww) > max_size_test:
        adj_scale = float(max_size_test) / max(newh, neww)
        newh = newh * adj_scale
        neww = neww * adj_scale

    # Round to nearest integer
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)

    # 4. Scaling factor for coordinate recovery
    scaling = max(newh / height, neww / width)

    # 5. Resize using OpenCV
    # Note: cv2.resize takes (width, height)
    resized_img = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)

    # 6. Create Canvas and Pad (Bottom-Right)
    # Create a blank canvas filled with the pad_color
    # NumPy uses (H, W, C)
    pad = np.full((canvas_h, canvas_w, 3), pad_color, dtype=np.uint8)

    # Paste the resized image into the top-left [0:height, 0:width]
    # We use min() to ensure we don't crash if the resized image exceeds canvas
    h_limit = min(newh, canvas_h)
    w_limit = min(neww, canvas_w)
    pad[0:h_limit, 0:w_limit] = resized_img[0:h_limit, 0:w_limit]

    return pad, scaling
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
def filter_prediction_exclusion_boxes(height,width,bbox,exclusion_boxes):
    # x,y,w,h=bbox
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[0] + bbox[2]
    y1 = bbox[1] + bbox[3]
    if exclusion_boxes is None or len(exclusion_boxes) < 2:
        exclusion_boxes = [(0, 0, 1, 1), (0, 0, 1, 1)]  # x0,y0,w,h
    if y1 <= height and x1 <= width and x0 >= 0 and y0 >= 0:
        cond1 = is_rectangle_overlapping(bbox, exclusion_boxes[0])
        cond2 = is_rectangle_overlapping(bbox, exclusion_boxes[1])
        if not cond1 and not cond2:
            return True
    return False