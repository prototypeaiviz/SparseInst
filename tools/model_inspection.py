from sparseinst import add_sparse_inst_config
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch.nn as nn


def print_model_structure():
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file("/home/mehran/Git-Thesis/SparseInst/configs/sparse_inst_r50_base.yaml")

    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 1
    model = build_model(cfg)

    print(f"{'Module Name':<50} | {'Type':<20} | {'Params':<10}")

    count_linear = 0
    count_conv = 0

    for name, module in model.named_modules():
        # Highlighting Conv2d and Linear layers as they are LoRA candidates
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            m_type = module.__class__.__name__

            # Simple heuristic to identify "important" layers
            if isinstance(module, nn.Conv2d):
                count_conv += 1
                if module.kernel_size == (1, 1):
                    pass
            elif isinstance(module, nn.Linear):
                count_linear += 1

            print(f"{name:<50} | {m_type:<20} | {params:<10}")

    print("-" * 90)
    print(f"Total Conv2d Layers: {count_conv}")
    print(f"Total Linear Layers: {count_linear}")

    print("\n\n=== Detailed View of Specific Components ===")
    print("\n--- Encoder (InstanceContextEncoder) ---")
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            print(f"Encoder 3x3 Conv: {name}")

    print("\n--- Decoder (GroupIAMDecoder) ---")
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            print(f"Decoder 3x3 Conv: {name}")

    print('-' * 90)
    print(model.encoder)
    print('-' * 90)
    print(model.backbone)

    # for name, module in model.backbone.res5[2].named_modules():
    #     print(f"{name:<10} | {module.output_shape}")
    # print(model.backbone.res5[2])

if __name__ == "__main__":
    print_model_structure()