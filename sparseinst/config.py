# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN
'''
Below is the breakdown of each parameter, followed by the code block with descriptive comments added.
Key Components Overview
    Encoder: Processes backbone features using:
            1 FPN (Feature Pyramid Network) 
            2 PPM (Pyramid Pooling Module) to create a rich context.
    Decoder (IAM): The "Instance Activation Maps" (IAM) module.
                   Instead of dense sliding windows, it predicts a fixed set of sparse kernels and masks.
    Matcher: Uses bipartite matching (likely Hungarian matching) to assign predicted masks to ground truth objects.
    Criterion/Loss: Combines 
                    Classification,
                    Pixel-wise Binary Cross Entropy,
                    Dice Loss, 
                    and an Objectness score (IOU-aware).
'''

def add_sparse_inst_config(cfg):

    # --- Global Model Settings ---
    cfg.MODEL.DEVICE = 'cuda'  # Run on GPU
    cfg.MODEL.MASK_ON = True  # Enable mask prediction (instance segmentation)

    # [SparseInst Root]
    cfg.MODEL.SPARSE_INST = CN()

    # --- Inference Hyperparameters ---
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.005  # Minimum score for a prediction to be considered valid
    cfg.MODEL.SPARSE_INST.MASK_THRESHOLD = 0.45  # Probability threshold to binarize the soft mask
    cfg.MODEL.SPARSE_INST.MAX_DETECTIONS = 100  # Limit for number of objects returned per image

    # [Encoder] - Refines backbone features
    cfg.MODEL.SPARSE_INST.ENCODER = CN()
    cfg.MODEL.SPARSE_INST.ENCODER.NAME = "FPNPPMEncoder"  # Encoder type (FPN + Pyramid Pooling)
    cfg.MODEL.SPARSE_INST.ENCODER.NORM = ""  # Normalization type (e.g., "GN" for Group Norm)
    cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES = ["res3", "res4", "res5"]  # Backbone levels used as input
    cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS = 256  # Internal channel dimension for encoder

    # [Decoder] - Generates Instance Activation Maps (IAM) and masks
    cfg.MODEL.SPARSE_INST.DECODER = CN()
    cfg.MODEL.SPARSE_INST.DECODER.NAME = "BaseIAMDecoder"
    cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100  # Number of learned "queries" or potential instances
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 80  # Number of categories (e.g., COCO has 80)

    # kernels for mask features
    cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM = 128  # Dimension of the dynamic kernels used for masks
    # upsample factor for output masks
    cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR = 2.0  # Factor to resize masks back toward original resolution
    cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM = False  # Whether to output the Instance Activation Maps visually
    cfg.MODEL.SPARSE_INST.DECODER.GROUPS = 4  # Groups for group convolution in mask generation

    # decoder.inst_branch (Instance branch: predicts classes)
    cfg.MODEL.SPARSE_INST.DECODER.INST = CN()
    cfg.MODEL.SPARSE_INST.DECODER.INST.DIM = 256  # Channel dimension for the instance branch
    cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS = 4  # Number of conv layers in the instance branch

    # decoder.mask_branch (Mask branch: predicts the raw mask features)
    cfg.MODEL.SPARSE_INST.DECODER.MASK = CN()
    cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM = 256  # Channel dimension for the mask branch
    cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS = 4  # Number of conv layers in the mask branch

    # [Loss] - How the model is penalized during training
    cfg.MODEL.SPARSE_INST.LOSS = CN()
    cfg.MODEL.SPARSE_INST.LOSS.NAME = "SparseInstCriterion"
    cfg.MODEL.SPARSE_INST.LOSS.ITEMS = ("labels", "masks")  # Components calculated in the loss

    # loss weights (scalars to balance different loss components)
    cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT = 2.0  # Importance of classification accuracy
    cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT = 5.0  # Importance of pixel-wise BCE mask loss
    cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT = 2.0  # Importance of Dice loss (handles class imbalance)
    cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT = 1.0  # Importance of predicting if a mask is actually an object

    # [Matcher] - Assigns predictions to Ground Truth (Bipartite Matching)
    cfg.MODEL.SPARSE_INST.MATCHER = CN()
    cfg.MODEL.SPARSE_INST.MATCHER.NAME = "SparseInstMatcher"
    cfg.MODEL.SPARSE_INST.MATCHER.ALPHA = 0.8  # Weight for classification cost in matching
    cfg.MODEL.SPARSE_INST.MATCHER.BETA = 0.2  # Weight for mask/segmentation cost in matching

    # [Optimizer] - Training solver settings
    cfg.SOLVER.OPTIMIZER = "ADAMW"  # Use AdamW optimizer
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0  # Learning rate multiplier for the backbone (1.0 = same as head)
    cfg.SOLVER.AMSGRAD = False  # Whether to use the AMSGrad variant of Adam

    # [Dataset mapper]
    cfg.MODEL.SPARSE_INST.DATASET_MAPPER = "SparseInstDatasetMapper"  # Custom logic for loading/augmenting data

    # [Pyramid Vision Transformer] - Config for PVT backbone if used
    cfg.MODEL.PVT = CN()
    cfg.MODEL.PVT.NAME = "b1"  # Model size/variant (b1)
    cfg.MODEL.PVT.OUT_FEATURES = ["p2", "p3", "p4"]  # Feature levels extracted from PVT
    cfg.MODEL.PVT.LINEAR = False  # Use spatial reduction attention (False = standard)

    # [CSPNet] - Config for CSP-Darknet (YOLO-style) backbones
    cfg.MODEL.CSPNET = CN()
    cfg.MODEL.CSPNET.NAME = "darknet53"  # Architecture name
    cfg.MODEL.CSPNET.NORM = ""  # Normalization type
    cfg.MODEL.CSPNET.OUT_FEATURES = ["csp1", "csp2", "csp3", "csp4"]  # Feature levels

    # [LoRA] - Low-Rank Adaptation Config
    cfg.MODEL.LORA = CN()
    cfg.MODEL.LORA.ENABLED = False
    cfg.MODEL.LORA.RANK = 8
    cfg.MODEL.LORA.ALPHA = 16.0

    # Target modules
    cfg.MODEL.LORA.BACKBONE = False
    cfg.MODEL.LORA.ENCODER = False
    cfg.MODEL.LORA.DECODER = False