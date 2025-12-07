
"""
Helper for running a quick compression evaluation (e.g., for Q2/Q3).

Typical usage in a Python shell:

    from compress_eval import run_q2
    run_q2(wbits=8, abits=8, calib_batches=8, epochs=30)
"""

import torch
from seed_utils import DEVICE, set_seed
from data_cifar10 import get_cifar10_loaders
from quantization_utils import (
    save_quant_weights,
    estimate_weight_bytes_incl_overheads,
    calibrate_activations,
    evaluate_with_output_fq,
)
from train_baseline import train_baseline, evaluate


def run_q2(
    wbits=8,
    abits=8,
    calib_batches=8,
    epochs=30,
    lr=0.2,
    wd=5e-4,
    batch_size=128,
):
    set_seed(42)
    # Train baseline (you can reduce epochs for debugging)
    model, _ = train_baseline(
        epochs=epochs,
        lr=lr,
        wd=wd,
        batch_size=batch_size,
    )

    # Compute FP32 bytes
    fp32_bytes = sum(
        p.numel() * 4
        for p in model.state_dict().values()
        if p.dtype.is_floating_point
    )

    # Save quantized weights
    npz_path, jsn_path, meta = save_quant_weights(
        model, "checkpoints", "q2", wbits
    )
    est_bytes = estimate_weight_bytes_incl_overheads(
        model, wbits, meta["items"]
    )

    print("========================================================")
    print("FP32 weights (bytes):", fp32_bytes)
    print("Quantized est bytes (incl overhead):", est_bytes)
    print("Compression ratio (weights):", fp32_bytes / est_bytes)
    print("Saved quantized weights to:", npz_path)
    print("Metadata JSON:", jsn_path)

    # Data loaders for calibration & test
    trainloader, valloader, testloader = get_cifar10_loaders(
        batch_size=batch_size
    )

    # Calibrate activations
    act_table = calibrate_activations(
        model, trainloader, max_batches=calib_batches, device=DEVICE
    )

    # Evaluate compressed model
    _, acc_c = evaluate_with_output_fq(
        model, testloader, act_table, abits, DEVICE
    )
    print(
        f"Compressed eval acc with {wbits}-bit weights and "
        f"{abits}-bit activations: {acc_c:.2f}%"
    )
    print("========================================================")

    return {
        "fp32_bytes": fp32_bytes,
        "quant_bytes_est": est_bytes,
        "compression_ratio": fp32_bytes / est_bytes,
        "compressed_acc": acc_c,
    }


if __name__ == "__main__":
    # Quick sanity check run (you can adjust epochs down for speed)
    run_q2(wbits=8, abits=8, calib_batches=4, epochs=30)
