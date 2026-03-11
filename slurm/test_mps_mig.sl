#!/bin/bash
#SBATCH -J "test_mps"
#SBATCH -o slurm/logs/test_mps_mig.out
#SBATCH -e slurm/logs/test_mps_mig.err
#SBATCH -p ar_mig
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 8G
#SBATCH --time=00:05:00

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== GPU info ==="
nvidia-smi

echo "=== Starting MPS ==="
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
nvidia-cuda-mps-control -d
MPS_EXIT=$?
echo "MPS start exit code: $MPS_EXIT"

echo "=== Running 2 parallel GPU workers ==="
python -u -c "
import torch, time
device = torch.device('cuda')
x = torch.randn(1, 3, 224, 224, device=device)
model = torch.hub.load('pytorch/vision', 'resnet18', weights='IMAGENET1K_V1').to(device).eval()
for i in range(50):
    with torch.no_grad():
        model(x)
print('Worker A done')
" &

python -u -c "
import torch, time
device = torch.device('cuda')
x = torch.randn(1, 3, 224, 224, device=device)
model = torch.hub.load('pytorch/vision', 'resnet18', weights='IMAGENET1K_V1').to(device).eval()
for i in range(50):
    with torch.no_grad():
        model(x)
print('Worker B done')
" &

wait
echo "=== Both workers finished ==="

echo quit | nvidia-cuda-mps-control
echo "=== MPS stopped ==="
