import json
import numpy as np
import cv2
import os
from utils_cv import  filter_prediction_exclusion_boxes

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

def prepare_gn_annotations(image,
                           exclusion_boxes_modified):
    gt_mask_cache = {}
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for ann_id,ann in enumerate(image.annotations):
        for poly_id, poly in enumerate(ann.segmentation):
            # Reshape coordinates to (N, 1, 2)
            poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(poly_np)

            bbox_xywh = [x, y, w, h]
            if not filter_prediction_exclusion_boxes(image.height, image.width, bbox_xywh,
                                                     exclusion_boxes_modified):
                continue
            mask_instance = np.zeros((image.height, image.width), dtype=np.uint8)

            # 3. Fill the polygon with white (255)
            # This creates a solid mask for IoU calculation
            cv2.fillPoly(mask, [poly_np], color=255)
            cv2.fillPoly(mask_instance, [poly_np], color=255)
            gt_mask_cache[ann_id] = mask_instance
    return mask,gt_mask_cache