#!/bin/bash
#SBATCH --job-name=2
#SBATCH --time=36:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/8.0
module load Anaconda3
python main_glassliquid.py -eta_initial=1e-3 -eta_final=1e-4 -eta_threshold=0.90 -batch_size=80 -iterations=1500 -beta=0.01 -keep_probability=0.5 -filename="Commit-test-1" -training_data_aug=True -restart=False -evaluation=False -restart_file="test.ckpt"



