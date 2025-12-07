
import argparse
import os
import torch

from seed_utils import set_seed, DEVICE
from data_cifar10 import get_cifar10_loaders
from model_mobilenetv2 import mobilenet_v2_cifar10
from quantization_utils import (
    save_quant_weights,
    estimate_weight_bytes_incl_overheads,
    calibrate_activations,
    evaluate_with_output_fq,
)
from train_baseline import evaluate


def load_model_from_ckpt(
    ckpt_path: str,
    num_classes: int = 10,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    stride1_stem: bool = True,
):
    """
    Build MobileNetV2 for CIFAR-10 and load the stored state_dict.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = mobilenet_v2_cifar10(
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
        stride1_stem=stride1_stem,
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Support both {"state_dict": ...} and direct state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="CS6886W A3 - Test MobileNetV2 with (optional) compression"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline_fp32.pt",
        help="Path to FP32 baseline checkpoint (.pt)",
    )
    parser.add_argument(
        "--weight_quant_bits",
        type=int,
        default=8,
        help="Bit-width for weight quantization (e.g. 8, 6, 4)",
    )
    parser.add_argument(
        "--activation_quant_bits",
        type=int,
        default=8,
        help="Bit-width for activation fake-quantization",
    )
    parser.add_argument(
        "--calib_batches",
        type=int,
        default=8,
        help="Number of batches from train set used for activation calibration",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation / loaders",
    )
    args = parser.parse_args()

    # -------------------- 0) Seed + data --------------------
    set_seed(42)
    trainloader, valloader, testloader = get_cifar10_loaders(
        batch_size=args.batch_size
    )

    # -------------------- 1) Load FP32 model ----------------
    model = load_model_from_ckpt(args.checkpoint)

    # FP32 weights size
    fp32_bytes = sum(
        p.numel() * 4
        for p in model.state_dict().values()
        if p.dtype.is_floating_point
    )

    # Baseline FP32 test accuracy
    test_loss, test_acc = evaluate(model, testloader)
    print("========================================================")
    print(f"[FP32] Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    print(
        f"[FP32] Total FP32 weight bytes: {fp32_bytes} B "
        f"({fp32_bytes / (1024**2):.3f} MB)"
    )
    print("========================================================")

    # -------------------- 2) Quantize weights ---------------
    os.makedirs("checkpoints", exist_ok=True)
    npz_path, jsn_path, meta = save_quant_weights(
        model, "checkpoints", "eval", args.weight_quant_bits
    )
    est_bytes = estimate_weight_bytes_incl_overheads(
        model, args.weight_quant_bits, meta["items"]
    )

    print(f"[Quant] Saved quantized weights to: {npz_path}")
    print(f"[Quant] Metadata JSON: {jsn_path}")
    print(
        f"[Quant] Estimated quantized weight bytes (incl. overhead): "
        f"{est_bytes} B ({est_bytes / (1024**2):.3f} MB)"
    )
    print(
        f"[Quant] Compression ratio (weights): "
        f"{fp32_bytes / est_bytes:.2f}x"
    )

    # -------------------- 3) Calibrate activations ----------
    print(
        "Calibrating activations on train set "
        f"({args.calib_batches} batches)..."
    )
    act_table = calibrate_activations(
        model,
        loader=trainloader,
        max_batches=args.calib_batches,
        device=DEVICE,
    )

    # -------------------- 4) Evaluate with fake-quant acts --
    _, acc_compressed = evaluate_with_output_fq(
        model,
        loader=testloader,
        act_max_table=act_table,
        abits=args.activation_quant_bits,
        device=DEVICE,
    )
    print("========================================================")
    print(
        f"[Quant] Test acc with "
        f"{args.weight_quant_bits}-bit weights & "
        f"{args.activation_quant_bits}-bit activations: "
        f"{acc_compressed:.2f}%"
    )
    print("========================================================")


if __name__ == "__main__":
    main()
