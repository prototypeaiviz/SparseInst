import os
import numpy as np
import cv2
import json
import onnxruntime as ort

class DatasetObject:
    def __init__(self, filepath):
        self.filepath = filepath

class Annotation:
    def __init__(self, ann_id, category_id, bbox, segmentation, area, iscrowd):
        self.id = ann_id
        self.category_id = category_id

        # COCO bbox: [x, y, w, h]
        self.bbox = np.array(bbox, dtype=np.float32)

        # segmentation: list of polygons (flattened)
        self.segmentation = segmentation  # keep raw, convert later if needed

        self.area = float(area)
        self.iscrowd = bool(iscrowd)

class Image(DatasetObject):
    def __init__(self, filepath):
        super().__init__(filepath)

        self.data = None
        self.channels = None
        self.height = None
        self.width = None
        self.filename = None

        # COCO-related
        self.image_id = None
        self.annotations = []   # List[Annotation]

    @staticmethod
    def get_image_size(image):
        if image is None:
            return -1, -1, -1

        if len(image.shape) == 3:
            h, w, c = image.shape
        elif len(image.shape) == 2:
            h, w = image.shape
            c = 1
        else:
            raise ValueError("Unexpected image shape.")

        return h, w, c

    def load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(self.filepath)

        self.filename = os.path.basename(self.filepath)
        self.data = cv2.imread(self.filepath, cv2.IMREAD_UNCHANGED)

        if self.data is None:
            raise ValueError(f"Failed to load image: {self.filepath}")

        self.height, self.width, self.channels = self.get_image_size(self.data)

def polygon_to_mask(segmentation, height, width):
    """
    Converts COCO polygon segmentation to a binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)

    return mask
def image_annotations_to_mask(image):
    """
    Builds a single binary mask for all annotations in an image.
    """
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for ann in image.annotations:
        if not ann.segmentation:
            continue

        ann_mask = polygon_to_mask(
            ann.segmentation,
            image.height,
            image.width
        )
        mask = np.maximum(mask, ann_mask)

    return mask

class CocoDataset:
    def __init__(self, images_dir, annotation_file):
        self.images_dir = images_dir
        self.annotation_file = annotation_file

        self.images = []              # List[Image]
        self.images_by_id = {}        # image_id -> Image
        self.categories = {}          # category_id -> name

        self._load()

    def _load(self):
        with open(self.annotation_file, "r") as f:
            coco = json.load(f)

        # -------------------------
        # Categories
        # -------------------------
        for cat in coco.get("categories", []):
            self.categories[cat["id"]] = cat["name"]

        # -------------------------
        # Images
        # -------------------------
        for img in coco["images"]:
            filepath = os.path.join(self.images_dir, img["file_name"])

            image = Image(filepath)
            image.image_id = img["id"]
            image.filename = img["file_name"]
            image.height = img.get("height")
            image.width = img.get("width")

            self.images.append(image)
            self.images_by_id[img["id"]] = image

        # -------------------------
        # Annotations
        # -------------------------
        for ann in coco["annotations"]:
            image_id = ann["image_id"]

            annotation = Annotation(
                ann_id=ann["id"],
                category_id=ann["category_id"],
                bbox=ann["bbox"],
                segmentation=ann.get("segmentation", []),
                area=ann.get("area", 0.0),
                iscrowd=ann.get("iscrowd", 0),
            )

            if image_id not in self.images_by_id:
                continue

            self.images_by_id[image_id].annotations.append(annotation)
def show_image_and_mask_opencv(image, mask, window_name="Image | Mask"):
    if image.data is None:
        image.load_data()

    # Ensure mask is uint8 and visible
    mask_vis = (mask * 255).astype(np.uint8)

    # Convert image to BGR if needed (cv2.imread already BGR)
    img_vis = image.data.copy()

    # If grayscale image, convert to BGR for stacking
    if len(img_vis.shape) == 2 or img_vis.shape[2] == 1:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    # Convert mask to 3 channels so we can concatenate
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    # Resize mask if needed (safety)
    if mask_vis.shape[:2] != img_vis.shape[:2]:
        mask_vis = cv2.resize(
            mask_vis,
            (img_vis.shape[1], img_vis.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Concatenate horizontally
    combined = np.hstack((img_vis, mask_vis))

    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_and_pad(img, short_length=640, long_length=1024):
    h, w = img.shape[:2]

    # 1. Calculate Resize Ratio
    # We want to scale so the short side is 'short_length'
    # but the long side doesn't exceed 'long_length'
    scale = short_length / min(h, w)
    if max(h, w) * scale > long_length:
        scale = long_length / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # 2. Perform Resize
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3. Calculate Padding to make it divisible by 32
    # This prevents the 83 vs 84 mismatch error
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32

    # Pad with zeros (bottom and right)
    padded_img = cv2.copyMakeBorder(resized_img, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_CONSTANT, value=0)

    return padded_img, (h, w)


def visualize_results(orig_img, masks, scores , threshold=0.5):
    """
    orig_img: The original BGR image (numpy array)
    masks: The mask output from ONNX (N, H, W)
    scores: The confidence scores
    """
    vis_img = orig_img.copy().transpose(1,2,0).astype(np.uint8)
    h, w = vis_img.shape[:2]
    scores = np.array(scores)
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue

        # 1. Get the mask and resize to original image dimensions
        mask = masks[i]
        # If mask is float, convert to binary (0 or 255)
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Generate a random color for this instance
        color = np.random.randint(0, 255, (3,)).tolist()

        # 3. Create a colored overlay
        colored_mask = np.zeros_like(vis_img, dtype=np.uint8)
        colored_mask[mask > 0] = color

        # 4. Blend the colored mask with the original image
        # Alpha is the transparency (0.5 = 50% transparent)
        vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

        # 5. Optional: Draw a border around the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, 2)

        # 6. Add label and score text
        text = f"ID:{int(scores[i]*100)}"
        cv2.putText(vis_img, text, (contours[0][0][0][0], contours[0][0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis_img


def postprocess_masks(masks, original_h, original_w, resized_h, resized_w):
    """
    masks: The raw output masks from ONNX (N, 640, 640)
    original_h/w: The dimensions of img.data
    resized_h/w: The dimensions (h, w) from your resize_img function
    """
    # 1. Remove the padding (slice the container)
    cropped_masks = masks[:, :resized_h, :resized_w]

    final_masks = []
    for mask in cropped_masks:
        # 2. Resize back to original dimensions
        # Use INTER_NEAREST for masks to avoid interpolating new label values
        original_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        final_masks.append(original_mask)

    return np.array(final_masks)
def main():
    test_json = "/media/aiviz05/New Volume/Data/TISSMART/Detectron/cvat-benchmark/cvat-output-coco.json"
    test_images = "/media/aiviz05/New Volume/Data/TISSMART/Detectron/cvat-benchmark/images"
    dataset = CocoDataset(
        images_dir=test_images,
        annotation_file=test_json
    )
    img = dataset.images[0]
    # Lazy load
    img.load_data()

    print(img.filename)
    print(img.data.shape)
    print("Num annotations:", len(img.annotations))
    mask = image_annotations_to_mask(img)
    # show_image_and_mask_opencv(img, mask)

    ann = img.annotations[0]
    print("BBox:", ann.bbox)
    print("Category:", dataset.categories[ann.category_id])
    # 1. Define ImageNet constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    providers = ['CUDAExecutionProvider', ]
    session_options = ort.SessionOptions()
    folder_containing_onnx_json= "/home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base_longrun/model.onnx"
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(folder_containing_onnx_json,providers=providers)
    resized_image, image_size = resize_and_pad(img.data, short_length=640, long_length=853)
    image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image = (image).astype('float32')
    # image = (image - mean) / std
    h, w = resized_image.shape[:2]
    # container = np.zeros((640, 640, 3), dtype=np.float32)
    # container[:h, :w, :] = image
    image = image.transpose( 2, 0, 1)
    plotting_image= image.copy()
    image = np.expand_dims(image,0)
    # 1. Prepare the input name (usually 'input' or 'images')
    input_name = ort_session.get_inputs()[0].name

    # 2. Run inference
    outputs = ort_session.run(
        None,  # Returns all output nodes
        {input_name: image}
    )
    # 3. Parse outputs (Assuming SparseInst standard output: labels, scores, masks)
    # Note: Check your model's specific output order/names if this varies
    scores = outputs[0][0]
    masks = outputs[1][0]

    result_image = visualize_results(plotting_image, masks, scores)
    print(result_image)



if __name__ == "__main__":
    main()