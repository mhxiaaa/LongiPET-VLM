#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu_b200
#SBATCH --gpus=b200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=15G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load miniconda
conda activate Med3DVLM
module load CUDA/12.6.0

python3 /home/mx79/project_pi_cl598/mx79/LongiPET-VLM-code/eval/eval_refSEG.py
