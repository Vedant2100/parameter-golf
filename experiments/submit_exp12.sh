#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000ada
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=exp
#SBATCH --output=logs/slurm_exp12_%j.out

source ~/.bashrc
conda activate pmgolf_stable
cd /data/AmitRoyChowdhury/vedant/parameter-golf

MAX_WALLCLOCK_SECONDS=21600 WEIGHT_DECAY=0.0 QK_GAIN_INIT=5.0 ./run_experiments.sh 12
