#!/bin/bash
#
#SBATCH --job-name=embed_cafa5
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=1600MB
#SBATCH --time=00:02:00
#SBATCH --partition=gpu,basic
#SBATCH --gres=shard:2
#SBATCH --constraint=t4

set -e

ml python3/3.12
source "$HOME/environments/python/esm-finetune/bin/activate"

DATA_DIR=$1

embed_cafa5 --data-dir "$DATA_DIR" --model-name "esm2_t6_8M_UR50D" --cpu "$SLURM_CPUS_PER_TASK"
