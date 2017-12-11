#!/bin/bash
#SBATCH --job-name=2
#SBATCH --time=36:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/8.0
module load Anaconda3
python main_glassliquid2.py -eta=1e-3 -batch_size=80 -epochs=1200 -beta=0.01 -keep_probability=0.5



