import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances, BitMasks
sys.path.append(".")
from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config
device = torch.device('cuda:0')
dtype = torch.float32
pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)

@torch.jit.script
def normalizer(x, mean, std): return (x - mean) / std


def synchronize():
    torch.cuda.synchronize()


def process_batched_inputs(batched_inputs):
    images = [x["image"].to(device) for x in batched_inputs]
    images = [normalizer(x, pixel_mean, pixel_std) for x in images]
    images = ImageList.from_tensors(images, 32)
    ori_size = (batched_inputs[0]["height"], batched_inputs[0]["width"])
    return images.tensor, images.image_sizes[0], ori_size


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    '''
    In many object detection models,
    a box might have a high classification score (e.g., "95% sure this is a cat") but the generated mask might be blurry
    or only cover half the cat.
    This function enforces a penalty:
        If the mask is perfectly aligned,
        the ratio is $\approx 1.0$, and the score remains high.
        If the mask is low quality or covers empty space,
        the ratio is $\approx 0.0$, and the final score is pushed toward zero


    '''
    # Intersection: (masks * mask_pred_).sum([1, 2])
    # calculates the sum of pixels where both the ground truth (or target) and the prediction overlap.
    mask_pred_ = mask_pred.float()
    # Normalization: It divides that intersection by the total area of the predicted mask:
    # mask_pred_.sum([1, 2]) + 1e-6. (The 1e-6 is a "epsilon" to prevent division by zero).
    # The Rescore: Finally, it multiplies the original scores by this ratio.
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


class SparseInst(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        output_shape = self.backbone.output_shape()

        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)

        self.to(self.device)

        # inference
        self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

    def forward(self, image, resized_size, ori_size):
        max_size = image.shape[2:]
        features = self.backbone(image)
        features = self.encoder(features)
        output = self.decoder(features)
        result = self.inference_single(
            output, resized_size, max_size, ori_size)
        return result

    def inference_single(self, outputs, img_shape, pad_shape, ori_shape):
        """
        inference for only one sample
        Args:
            scores (tensor): [NxC]
            masks (tensor): [NxHxW]
            img_shape (list): (h1, w1), image after resized
            pad_shape (list): (h2, w2), padded resized image
            ori_shape (list): (h3, w3), original shape h3*w3 < h1*w1 < h2*w2
        """
        result = Instances(ori_shape)
        # scoring
        pred_logits = outputs["pred_logits"][0].sigmoid()
        pred_scores = outputs["pred_scores"][0].sigmoid().squeeze()
        pred_masks = outputs["pred_masks"][0].sigmoid()
        # obtain scores
        scores, labels = pred_logits.max(dim=-1)
        # remove by thresholding
        keep = scores > self.cls_threshold
        scores = torch.sqrt(scores[keep] * pred_scores[keep])
        labels = labels[keep]
        pred_masks = pred_masks[keep]

        if scores.size(0) == 0:
            return None
        scores = rescoring_mask(scores, pred_masks > 0.45, pred_masks)
        # it is comparing a "hard" binary mask against a "soft" probability mask.
        # The "Mask Sparsity" PenaltyBy passing pred_masks > 0.45 as the mask
        # and pred_masks (the raw sigmoid probabilities) as the "target,"
        # the function measures how confident the model is about the pixels it just claimed were part of the object.
            # High Confidence: If the pixels inside the mask have values close to $1.0$, the ratio stays high.
            # Low Confidence (Uncertainty): If the pixels are all hovering around $0.46$ (just barely passing the threshold),
        # the sum of the raw probabilities will be much lower than the count of pixels.
        # This scales the score down because the model is "unsure."
        h, w = img_shape
        # resize masks
        pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=pad_shape,
                                   mode="bilinear", align_corners=False)[:, :, :h, :w]
        pred_masks = F.interpolate(pred_masks,
                                   size=ori_shape,
                                   mode='bilinear',
                                   align_corners=False).squeeze(1)
        mask_pred = pred_masks > self.mask_threshold
        mask_pred = BitMasks(mask_pred)
        result.pred_masks = mask_pred
        result.scores = scores
        result.pred_classes = labels
        return result