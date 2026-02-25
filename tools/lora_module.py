"""
LoRA (Low-Rank Adaptation) for SparseInst
"""

from typing import Iterable, Tuple, Optional, Dict
from sparseinst import add_sparse_inst_config
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch.nn as nn
import torch
import math


class LoRAConv2d(nn.Module):
    """
    LoRA adapter wrapper for nn.Conv2d layers.

    Wraps an existing Conv2d layer with low-rank trainable adapters while keeping
    the original weights frozen. The forward pass computes:
        output = base_conv(x) + scaling * lora_B(lora_A(x))

    where lora_A and lora_B are 1×1 convolutions forming the low-rank decomposition.

    Args:
        conv (nn.Conv2d): The base Conv2d layer to wrap (will be frozen)
        rank (int): Rank of the low-rank decomposition. Higher rank = more capacity
                    but more parameters. Typical values: 4, 8, 16. Default: 4
        alpha (float): Scaling factor for LoRA output. Effective scaling = alpha/rank.
                      Higher alpha = stronger adaptation. Default: 1.0

    Attributes:
        base_conv (nn.Conv2d): Original frozen Conv2d layer
        lora_A (nn.Conv2d): Down-projection: in_channels → rank
        lora_B (nn.Conv2d): Up-projection: rank → out_channels
        rank (int): LoRA rank
        alpha (float): Scaling factor
        scaling (float): Computed as alpha / rank

    Note:
        - lora_A is initialized with Kaiming normal (similar to base conv)
        - lora_B is initialized to zeros (so initially LoRA has no effect)
        - This ensures training starts from the pretrained base weights
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super(LoRAConv2d, self).__init__()

        assert isinstance(conv, nn.Conv2d), "LoRAConv2d expects a nn.Conv2d as base layer"

        # Store and freeze the base convolution
        self.base_conv: nn.Conv2d = conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

        # Extract base conv properties
        in_channels = self.base_conv.in_channels
        out_channels = self.base_conv.out_channels
        self.kernel_size = self.base_conv.kernel_size
        self.stride = self.base_conv.stride
        self.padding = self.base_conv.padding
        self.dilation = self.base_conv.dilation
        self.groups = self.base_conv.groups
        self.bias = self.base_conv.bias is not None

        # LoRA hyperparameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(self.alpha) / max(1, self.rank)

        # Down-projection: in_channels -> rank (using 1x1 conv)
        self.lora_A = nn.Conv2d(
            in_channels,
            rank,
            kernel_size=1,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=False
        )

        # Up-projection: rank -> out_channels (using 1x1 conv)
        self.lora_B = nn.Conv2d(
            rank,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Initialize A with small random values, B with zeros
        # This ensures LoRA starts with no effect
        nn.init.kaiming_normal_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # This ensures compatibility with AMP and multi-GPU training
        if self.base_conv.weight.device.type != 'cpu':
            self.lora_A = self.lora_A.to(self.base_conv.weight.device)
            self.lora_B = self.lora_B.to(self.base_conv.weight.device)

        # Match dtype to base_conv for AMP compatibility
        self.lora_A = self.lora_A.to(self.base_conv.weight.dtype)
        self.lora_B = self.lora_B.to(self.base_conv.weight.dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base convolution with LoRA adaptation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Output of shape (B, out_channels, H', W')
                         where H' and W' depend on kernel_size, stride, padding
        """
        # Base convolution (frozen weights)
        base_out = self.base_conv(x)

        lora_out = self.lora_B(self.lora_A(x)) * self.scaling

        # Combine base and LoRA outputs
        return base_out + lora_out

    def merge_into_base(self) -> None:
        """
        Merge LoRA adapter weights into the base convolution weights.

        This allows deployment with standard Conv2d layers (no LoRA wrapper needed).
        The merged weight effectively becomes: W_new = W_base + (alpha/rank) * B @ A

        Warning:
            After merging, the LoRA effect is baked into base_conv. Don't call this
            during training, only for deployment/inference.

        """
        with torch.no_grad():
            # base_conv.weight shape: (out_channels, in_channels, kh, kw)
            W = self.base_conv.weight
            out_ch, in_ch, kh, kw = W.shape

            # Extract LoRA matrices and reshape to 2D
            # lora_A.weight shape: (rank, in_channels, 1, 1) -> reshape to (rank, in_channels)
            # lora_B.weight shape: (out_channels, rank, 1, 1) -> reshape to (out_channels, rank)
            A = self.lora_A.weight.view(self.rank, in_ch)
            B = self.lora_B.weight.view(out_ch, self.rank)

            # Compute the low-rank update: deltaW = (alpha/rank) * B @ A
            # Shape: (out_channels, in_channels)
            Delta_2d = B.matmul(A) * self.scaling

            # Expand to match kernel spatial dimensions
            # For kh×kw kernels, place the update at the center position
            delta = torch.zeros_like(W)
            center_h = kh // 2
            center_w = kw // 2
            delta[:, :, center_h, center_w] = Delta_2d

            # Merge: add LoRA update to base weights in-place
            W.add_(delta)


def _recursive_replace_conv_with_lora(
    parent: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_class: type = nn.Conv2d,
    skip_names: Optional[Iterable[str]] = None
) -> int:
    """
    Recursively traverse a module and replace Conv2d layers with LoRA wrappers.

    This function walks through all child modules of `parent` and replaces instances
    of `target_class` (typically nn.Conv2d) with LoRAConv2d wrappers in-place.

    Args:
        parent (nn.Module): Parent module to traverse
        rank (int): LoRA rank for the adapters. Default: 4
        alpha (float): LoRA scaling factor. Default: 1.0
        target_class (type): Class to replace (typically nn.Conv2d). Default: nn.Conv2d
        skip_names (Optional[Iterable[str]]): Names of immediate children to skip.
                                              Default: None (replace all)

    Returns:
        int: Number of layers replaced

    Note:
        - Only immediate children names in `skip_names` are skipped
        - Recursion into grandchildren uses skip_names=None to avoid breaking structure
    """
    if skip_names is None:
        skip_names = set()
    else:
        skip_names = set(skip_names)

    replaced = 0
    for name, module in list(parent._modules.items()):
        # Skip specified module names
        if name in skip_names:
            continue

        # If this module is exactly the target class, replace it
        if isinstance(module, target_class) and (module.kernel_size == (3, 3) or module.kernel_size == 3):
            parent._modules[name] = LoRAConv2d(module, rank=rank, alpha=alpha)
            replaced += 1
        elif isinstance(module, nn.Module):
            # Recurse into child modules
            # Note: skip_names only applies to immediate children, so pass None for recursion
            replaced += _recursive_replace_conv_with_lora(
                module,
                rank=rank,
                alpha=alpha,
                target_class=target_class,
                skip_names=None  # Don't propagate skip_names to deeper levels
            )
    return replaced


def apply_lora_to_model(
    model: nn.Module,
    targets: Iterable[str] = ('encoder', 'decoder'),
    rank: int = 4,
    alpha: float = 1.0,
    skip: Optional[Dict[str, Iterable[str]]] = None
) -> Tuple[int, Dict[str, int]]:
    """
    Apply LoRA adapters to specific submodules of a model.

    Args:
        model (nn.Module): The model to modify
        targets (Iterable[str]): Names of top-level submodules to apply LoRA to.
                                 For SMP models: ['encoder', 'decoder']. Default: ('encoder', 'decoder')
        rank (int): LoRA rank. Higher = more capacity but more parameters.
                    Typical values: 4, 8, 16. Default: 4
        alpha (float): LoRA scaling factor.
        skip (Optional[Dict[str, Iterable[str]]]): Per-target skip lists.
                                                   Example: {'encoder': ['conv1', 'bn1']}
                                                   to skip first layer. Default: None

    Returns:
        Tuple[int, Dict[str, int]]:
            - Total number of Conv2d layers replaced across all targets
            - Dictionary mapping target name to number of replacements in that target
    Note:
        - Non-existent targets are silently skipped (count = 0)
        - After applying LoRA, call freeze_all_but_lora() to freeze base weights
    """
    total = 0
    per_target = {}

    for target_name in targets:
        # Check if model has this attribute
        if not hasattr(model, target_name):
            per_target[target_name] = 0
            continue

        # Get the submodule
        submodule = getattr(model, target_name)

        # Get skip list for this target (if any)
        skip_names = None
        if skip and target_name in skip:
            skip_names = skip[target_name]

        # Apply LoRA to this submodule
        num_replaced = _recursive_replace_conv_with_lora(
            submodule,
            rank=rank,
            alpha=alpha,
            skip_names=skip_names
        )

        per_target[target_name] = num_replaced
        total += num_replaced

    return total, per_target


def freeze_all_but_lora(model: nn.Module) -> None:
    """
    Freeze all model parameters except LoRA adapter parameters.

    Args:
        model (nn.Module): Model with LoRA adapters applied

    Note:
        - First freezes ALL parameters (requires_grad=False)
        - Then unfreezes only parameters within LoRAConv2d modules
        - BatchNorm should also be frozen via set_batchnorm_eval()
    """
    # First, freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Then, unfreeze LoRA adapter parameters
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for p in module.lora_B.parameters():
                p.requires_grad = True

    # Unfreeze task-specific modules that were skipped from LoRA
    # These modules should be trained fully for task adaptation
    for name, module in model.named_modules():
        # Unfreeze SparseInst prediction heads
        if any(head_name in name for head_name in ['iam_conv', 'cls_score', 'mask_kernel', 'objectness', 'projection']):
            for p in module.parameters():
                p.requires_grad = True

def set_batchnorm_eval(model: nn.Module) -> None:
    """
    Set all BatchNorm layers to eval mode and freeze their parameters.

    When fine-tuning with LoRA on small datasets, it's often beneficial to
    freeze BatchNorm statistics (running mean/var) from pretraining rather
    than updating them.

    Args:
        model (nn.Module): Model to modify

    Note:
        - This is called once before training starts
        - Even when model.train() is called, these BN layers stay in eval mode
        - Prevents BN statistics drift on small fine-tuning datasets
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and LoRA-specific parameters in a model.

    Useful for verifying that LoRA was applied correctly and understanding
    the parameter efficiency.

    Args:
        model (nn.Module): Model to analyze (with or without LoRA)

    Returns:
        Dict[str, int]: Dictionary with keys:
            - 'total': Total number of parameters in model
            - 'trainable': Number of trainable parameters (requires_grad=True)
            - 'lora_parameters': Number of parameters in LoRA adapters specifically
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count LoRA parameters specifically
    lora_parameters = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, LoRAConv2d)
        for p in m.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "lora_parameters": lora_parameters
    }


def merge_all_lora_into_base(model: nn.Module) -> int:
    """
    Merge all LoRA adapters into their base convolution weights.

    After training, this function adds the LoRA adaptations into the base
    Conv2d weights, preparing for deployment. The model will still have
    LoRAConv2d wrappers, but they contain the merged weights.

    Args:
        model (nn.Module): Model with LoRA adapters

    Returns:
        int: Number of LoRA modules merged

    Warning:
        - This modifies base_conv weights in-place
        - After merging, LoRA adapters still exist but have no effect
        - Use replace_lora_wrappers_with_base_conv() to remove wrappers entirely
    """
    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRAConv2d):
            module.merge_into_base()
            merged += 1
    return merged


def replace_lora_wrappers_with_base_conv(model: nn.Module) -> None:
    """
    Replace LoRAConv2d wrappers with their inner base_conv layers.

    After merging LoRA weights (see merge_all_lora_into_base), this function
    removes the LoRA wrappers entirely, leaving only standard Conv2d layers
    with the merged weights.

    Args:
        model (nn.Module): Model with merged LoRA adapters

    Note:
        - Must call merge_all_lora_into_base() first
        - After this, model has no LoRA components (standard PyTorch modules only)
        - Resulting model can be used with standard inference pipelines
    """
    def _replace(parent: nn.Module):
        """Recursively replace LoRAConv2d with base_conv"""
        for key, child in list(parent._modules.items()):
            if isinstance(child, LoRAConv2d):
                # Replace wrapper with inner base conv
                parent._modules[key] = child.base_conv
            elif isinstance(child, nn.Module):
                # Recurse into child modules
                _replace(child)

    _replace(model)


def apply_lora_from_config(model, cfg):
    """
    Apply LoRA to the model based on Detectron2 configuration.
    Wraps apply_lora_to_model.
    """
    if not cfg.MODEL.LORA.ENABLED:
        return model
        
    rank = cfg.MODEL.LORA.RANK
    alpha = cfg.MODEL.LORA.ALPHA
    
    targets = []
    # skips = {}        ### I need to add this part

    if cfg.MODEL.LORA.BACKBONE:
        targets.append("backbone")
    if cfg.MODEL.LORA.ENCODER:
        targets.append("encoder")
    if cfg.MODEL.LORA.DECODER:
        targets.append("decoder")

    # Add Skip to the code
    # if cfg.MODEL.LORA.SKIP:
    #     pass

    print(f"Applying LoRA with Rank={rank}, Alpha={alpha}...")
    print(f"Targets: {targets}")
    
    total, per_target = apply_lora_to_model(
        model,
        targets=targets,
        rank=rank,
        alpha=alpha,
        skip=None
    )
    
    print(f"Replaced {total} Conv2d layers.")
    for t, count in per_target.items():
        print(f"  - {t}: {count} layers")

    print("Freezing base parameters and BN...")
    freeze_all_but_lora(model)
    set_batchnorm_eval(model)
    
    stats = count_parameters(model)
    print(f"LoRA Applied. Trainable Params: {stats['trainable']:,} / {stats['total']:,} "
          f"({100 * stats['trainable'] / stats['total']:.2f}%)\n")
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Application Test: SparseInst with ResNet50 Encoder")
    print("=" * 60)

    cfg = get_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    add_sparse_inst_config(cfg)
    cfg.merge_from_file("/home/mehran/Git-Thesis/SparseInst/configs/sparse_inst_r50_base.yaml")
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 1
    model = build_model(cfg)

    # Count original parameters
    print("\n1. Original Model:")
    stats_original = count_parameters(model)
    print(f"   Total parameters: {stats_original['total']:,}")
    print(f"   Trainable parameters: {stats_original['trainable']:,}")

    # Define LoRA configuration
    targets = ['backbone', 'encoder', 'decoder']

    # Apply LoRA
    print("\n2. Applying LoRA:")
    print(f"   Rank: 8, Alpha: 16.0")
    print(f"   Targets: {targets}")
    # print(f"   Skip: {skip}")

    total, per_target = apply_lora_to_model(
        model,
        targets=targets,
        rank=8,
        alpha=16.0,
        skip=None
    )

    print(f"\n   Replaced {total} Conv2d layers:")
    for target, count in per_target.items():
        print(f"   - {target}: {count} layers")

    # Freeze base parameters
    print("\n3. Freezing base parameters...")
    freeze_all_but_lora(model)
    set_batchnorm_eval(model)

    # Count parameters after LoRA
    print("\n4. Model with LoRA:")
    stats_lora = count_parameters(model)
    print(f"   Total parameters: {stats_lora['total']:,}")
    print(f"   Trainable parameters: {stats_lora['trainable']:,}")
    print(f"   LoRA parameters: {stats_lora['lora_parameters']:,}")

    reduction = 100 * stats_lora['trainable'] / stats_lora['total']
    print(f"\n   Parameter efficiency: {reduction:.2f}% trainable")
    print(f"   Reduction factor: {stats_lora['total'] / stats_lora['trainable']:.1f}x")

    print("\n" + "=" * 60)
    print("LoRA application complete!")

    print(model)