
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

Seed Configuration

All runs use a fixed seed for reproducibility.
In seed_utils.py:

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



Question 5(b) – README and exact commands

Have uploased the trained baseline in checkpoints can directly use that and run this command 

python test.py --weight_quant_bits 8 --activation_quant_bits 8



Below are the steps which can be done from beginning in case:

Setup:

git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt


Train FP32 baseline (Q1)

python train_baseline.py


This trains MobileNet-V2 on CIFAR-10 for 30 epochs, plots the train/val accuracy curves, and saves the checkpoint to checkpoints/baseline_fp32.pt.

Evaluate FP32 baseline accuracy

python test.py --weight_quant_bits 32 --activation_quant_bits 32


Run a single compressed configuration (Q2/Q4)
Example: 4-bit weights, 8-bit activations:

python test.py --weight_quant_bits 4 --activation_quant_bits 8


Run the 8-run sweep used in Q3
(I expose this as a flag / function in the script):

python test.py --run_q3_sweep 1


This command reuses the baseline checkpoint, runs the 8 (wbits, abits) combinations, logs to WandB, and prints the table of compression_ratio, model_size_mb, and quantized_acc used in Q

Also can execute for 8-bit weights and 8-bit activations



