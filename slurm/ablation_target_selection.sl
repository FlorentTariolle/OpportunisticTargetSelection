#!/bin/bash
#SBATCH -J "ablation_tgt"
#SBATCH -o slurm/logs/ablation_tgt.out
#SBATCH -e slurm/logs/ablation_tgt.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=08:00:00

# Target-selection ablation: clean_argmax + random_target (400 runs)
# Submit:  sbatch slurm/ablation_target_selection.sl
# Resume:  sbatch again — CSV keys prevent duplicate work.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user -r requirements-hpc.txt
#   python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"

module purge
module load aidl/pytorch/2.6.0-cuda12.6

N_WORKERS=${N_WORKERS:-10}
N_IMAGES=100
CHUNK=$(( (N_IMAGES + N_WORKERS - 1) / N_WORKERS ))

# Start NVIDIA MPS to share a single GPU context across workers.
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
    python -u benchmarks/ablation_target_selection.py --image-start $START --image-end $END > /dev/null 2>&1 &
done
wait

# Stop MPS
echo quit | nvidia-cuda-mps-control
echo "All workers finished at $(date)"
