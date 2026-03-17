#!/bin/bash
#SBATCH -J "ablation_robust"
#SBATCH -o slurm/logs/ablation_naive_robust.out
#SBATCH -e slurm/logs/ablation_naive_robust.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 16G
#SBATCH --time=08:00:00

# Naive ablation on robust ResNet-50: T=100 vs OT (#23)
# Submit:  sbatch slurm/ablation_naive_robust.sl
# Monitor: bash slurm/monitor.sh results/benchmark_ablation_naive_robust.csv 400
# Resume:  sbatch again — CSV keys prevent duplicate work.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user -r requirements-hpc.txt

module purge
module load aidl/pytorch/2.6.0-cuda12.6

# Start NVIDIA MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
nvidia-cuda-mps-control -d
echo "MPS started"

N_WORKERS=${N_WORKERS:-10}
N_IMAGES=100
CHUNK=$(( (N_IMAGES + N_WORKERS - 1) / N_WORKERS ))

echo "Starting $N_WORKERS workers at $(date)"
for i in $(seq 0 $((N_WORKERS - 1))); do
    START=$((i * CHUNK))
    END=$(( (i + 1) * CHUNK ))
    [ $END -gt $N_IMAGES ] && END=$N_IMAGES
    [ $START -ge $N_IMAGES ] && continue
    python -u benchmarks/ablation_naive.py --source robust --image-start $START --image-end $END > /dev/null 2>&1 &
done
wait

# Stop MPS
echo quit | nvidia-cuda-mps-control
echo "All workers finished at $(date)"
