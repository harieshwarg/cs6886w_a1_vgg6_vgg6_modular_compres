
# CS6886W — Assignment 3
MobileNet-V2 Training + Compression (Quantization) on CIFAR-10

## 📁 Folder Structure

- `seed_utils.py` — Seed configuration & CUDA device helper
- `data_cifar10.py` — CIFAR-10 dataloaders with augmentation
- `model_mobilenetv2.py` — MobileNet-V2 modified for CIFAR-10
- `train_baseline.py` — Baseline FP32 training + evaluation (saves checkpoint)
- `quantization_utils.py` — Manual quantization (weights + activations)
- `compress_eval.py` — Helper to run compression experiment (Q2/Q3-style)
- `test.py` — Entry-point script used by evaluator (FP32 + compressed eval)
- `requirements.txt` — Environment dependencies
- `README.md` — This documentation

---
Execute

!python test.py --weight_quant_bits 8 --activation_quant_bits 8

