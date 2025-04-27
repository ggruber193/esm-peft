#!/bin/bash
#
#SBATCH --job-name=finetune_esm
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=1600MB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=t4

set -e

ml python3/3.12
source "$HOME/environments/python/esm-finetune/bin/activate"

DATA_DIR=$1

finetune_cafa5 --data-dir "$DATA_DIR" --model-name "esm2_t6_8M_UR50D" --batch-size 16 --cpu "$SLURM_CPUS_PER_TASK" --tmp "$TMPDIR"
