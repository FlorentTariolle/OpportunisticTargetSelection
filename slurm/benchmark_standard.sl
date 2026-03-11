#!/bin/bash
#SBATCH -J "bench_std"
#SBATCH -o slurm/logs/bench_std.out
#SBATCH -e slurm/logs/bench_std.err
#SBATCH -p ar_mig
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 8G
#SBATCH --time=08:00:00

# Full standard benchmark: 5 models x 3 methods x 100 images x 3 modes
# Submit:  sbatch slurm/benchmark_standard.sl
# Resume:  sbatch again — CSV keys prevent duplicate work.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user -r requirements-hpc.txt
#   python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"
#   python -c "import torchvision; torchvision.models.vit_b_16(weights='IMAGENET1K_V1')"

module purge
module load aidl/pytorch/2.6.0-cuda12.6

# Start NVIDIA MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
nvidia-cuda-mps-control -d
echo "MPS started"

echo "Starting benchmark at $(date)"
# Kill workers after 7h55m so the requeue block has time to run before 8h wall time
timeout 28500 python -u benchmark.py --part 1 --n-images 100 --source standard > /dev/null 2>&1 &
timeout 28500 python -u benchmark.py --part 2 --n-images 100 --source standard > /dev/null 2>&1 &
wait

# Stop MPS
echo quit | nvidia-cuda-mps-control
echo "All workers finished at $(date)"

# Auto-requeue if there's still work to do
TOTAL=$(wc -l < results/benchmark_standard.csv)
if [ "$TOTAL" -lt 4501 ]; then
    echo "Only $((TOTAL - 1))/4500 done, resubmitting..."
    sbatch slurm/benchmark_standard.sl
fi
