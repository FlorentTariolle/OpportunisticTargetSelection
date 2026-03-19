#!/bin/bash
#SBATCH -J "multiseed"
#SBATCH -o slurm/logs/multiseed.out
#SBATCH -e slurm/logs/multiseed.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Multi-seed validation (#24): 2 methods x 100 images x 5 seeds x 3 modes = 3,000 runs
# Submit:  sbatch slurm/multiseed.sl
# Resume:  sbatch again — CSV keys prevent duplicate work.
# Expected: ~3 submissions to complete all 3,000 runs.

module purge
module load aidl/pytorch/2.6.0-cuda12.6

N_WORKERS=${N_WORKERS:-10}
N_IMAGES=100
CHUNK=$(( (N_IMAGES + N_WORKERS - 1) / N_WORKERS ))

# Start NVIDIA MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
nvidia-cuda-mps-control -d
echo "MPS started"

echo "Starting $N_WORKERS workers at $(date)"
for i in $(seq 0 $((N_WORKERS - 1))); do
    START=$((i * CHUNK))
    END=$(( (i + 1) * CHUNK ))
    [ $END -gt $N_IMAGES ] && END=$N_IMAGES
    [ $START -ge $N_IMAGES ] && continue
    python -u benchmarks/multiseed.py --image-start $START --image-end $END > slurm/logs/worker_${i}.err 2>&1 &
done
wait

# Stop MPS
echo quit | nvidia-cuda-mps-control
echo "All workers finished at $(date)"

# Auto-requeue if work remains (3,000 data rows + 1 header = 3,001 lines)
EXPECTED=3001
TOTAL=$(wc -l < results/benchmark_multiseed.csv 2>/dev/null || echo 0)
if [ "$TOTAL" -lt "$EXPECTED" ]; then
    echo "Only $((TOTAL - 1))/3000 done, resubmitting..."
    sbatch slurm/multiseed.sl
else
    echo "All 3,000 runs complete."
fi
