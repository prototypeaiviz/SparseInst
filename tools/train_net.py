from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from typing import Any, Dict, List, Set
from detectron2.config import get_cfg
import detectron2.utils.comm as comm
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from datetime import datetime
import itertools
import torch
import sys
import os

sys.path.append(".")
from sparseinst import add_sparse_inst_config, COCOMaskEvaluator
from lora_module import apply_lora_from_config, remap_checkpoint_for_lora
from detectron2.checkpoint import DetectionCheckpointer
import wandb



class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        """Build the model and apply LoRA before training starts."""
        model = DefaultTrainer.build_model(cfg)
        if cfg.MODEL.LORA.ENABLED:
            model = apply_lora_from_config(model, cfg)
        return model

    def resume_or_load(self, resume=True):
        """
        Two-pass LoRA-aware checkpoint loading.

        Pass 1: Use Detectron2's standard DetectionCheckpointer. It handles
                all checkpoint formats (pkl, caffe2, DDP 'module.' prefix, ndarray
                conversion, resume-from-last-saved-checkpoint logic). Encoder/decoder
                weights in LoRA-wrapped layers will be silently skipped — expected.

        Pass 2: Do a targeted second load to fill the `base_conv` weights for
                layers that Pass 1 skipped because of the LoRA key name change
                (e.g. checkpoint: 'encoder.fpn_outputs.0.weight'
                       model expects: 'encoder.fpn_outputs.0.base_conv.weight').

        Without Pass 1, raw `torch.load()` misses pkl/caffe2 format handling,
        DDP prefix stripping, and resume logic — causing random initialization
        and loss ≈ 200 on the very first iteration.
        """
        # Pass 1 — standard Detectron2 loading
        super().resume_or_load(resume=resume)

        if not self.cfg.MODEL.LORA.ENABLED:
            return  # nothing LoRA-specific to do

        # Determine which checkpoint file was actually loaded
        # (could be last checkpoint in OUTPUT_DIR when resume=True)
        last_ckpt = self.checkpointer.get_checkpoint_file()
        checkpoint_path = last_ckpt if (resume and last_ckpt) else self.cfg.MODEL.WEIGHTS

        if not checkpoint_path:
            return

        print("LoRA Pass 2: filling base_conv weights for LoRA-wrapped layers...")
        try:
            raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            print(f"  WARNING: Could not load file for LoRA Pass 2: {e}")
            return

        state_dict = raw_ckpt.get("model", raw_ckpt)

        # Strip "module." prefix if saved from DDP training
        state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                      for k, v in state_dict.items()}

        model_keys = set(self.model.state_dict().keys())
        params_buffers = dict(
            list(self.model.named_parameters()) +
            list(self.model.named_buffers())
        )

        loaded_count = 0
        with torch.no_grad():
            for ckpt_key, value in state_dict.items():
                if ckpt_key in model_keys:
                    continue   # already loaded by Pass 1
                parts = ckpt_key.rsplit(".", 1)
                if len(parts) != 2:
                    continue
                parent_path, param_name = parts
                candidate = f"{parent_path}.base_conv.{param_name}"
                if candidate in model_keys and candidate in params_buffers:
                    target = params_buffers[candidate]
                    params_buffers[candidate].copy_(
                        torch.as_tensor(value).to(target.device, target.dtype)
                    )
                    loaded_count += 1

        print(f"  LoRA Pass 2 complete: loaded {loaded_count} base_conv tensors.")


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        MetadataCatalog.get(dataset_name).set(thing_classes=["Pill"])
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOMaskEvaluator(dataset_name, ("segm", ), True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            # for transformer
            if "patch_embed" in key or "cls_token" in key:
                weight_decay = 0.0
            if "norm" in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full  model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + f"_Dual_LORA_No_backbone_APRIL_2{timestamp}"
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "sparseinst" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sparseinst")
    return cfg


def main(args):

    # AIVIZ-05 config paths:

    # register_coco_instances(
    #     "pills_train", {},
    #     f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/SparseInst_ALL_DUAL/labelings/train.json",
    #     f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/SparseInst_ALL_DUAL/data/train/imgs"
    # )
    #
    # register_coco_instances(
    #     "pills_val", {},
    #     f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/SparseInst_ALL_DUAL/labelings/val.json",
    #     f"/media/aiviz05/New Volume/Data/TISSMART/Detectron/SparseInst_ALL_DUAL/data/val/imgs"
    # )

    # Personal paths
    register_coco_instances(
        "pills_train", {},
        f"/home/mehran/Desktop/Stuff/SparseInst_ALL_DUAL/train.json",
        f"/home/mehran/Desktop/Stuff/SparseInst_ALL_DUAL/train/imgs"
    )

    register_coco_instances(
        "pills_val", {},
        f"/home/mehran/Desktop/Stuff/SparseInst_ALL_DUAL/val.json",
        f"/home/mehran/Desktop/Stuff/SparseInst_ALL_DUAL/val/imgs"
    )

    cfg = setup(args)
    wandb.init(
        project="SparseInst",
        name="Dual_LORA_No_backbone_APRIL_2",
        config=cfg,
        sync_tensorboard=True
    )
    if args.eval_only:
        model = Trainer.build_model(cfg)
        if cfg.MODEL.LORA.ENABLED:
            # The checkpoint was saved by a LoRA model, but if we ever load a
            # non-LoRA checkpoint for eval we still need the remapping.
            # Use the same model-aware remapping as the training path.
            print("LoRA enabled: remapping checkpoint keys for eval...")
            raw_ckpt = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
            state_dict = raw_ckpt.get("model", raw_ckpt)
            remapped = remap_checkpoint_for_lora(model, state_dict)
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            real_missing = [k for k in missing if "lora_" not in k]
            real_unexpected = [k for k in unexpected if "lora_" not in k]
            if real_missing:
                print(f"WARNING: Truly missing (non-LoRA) keys: {real_missing}")
            if real_unexpected:
                print(f"WARNING: Truly unexpected (non-LoRA) keys: {real_unexpected}")
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )