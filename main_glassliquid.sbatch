#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --time=36:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output
#SBATCH --error=errors

module load singularity
/software/src/singularity_images/tensorflow_1.1.0.img 367.44 3 main_glassliquid.py -eta_initial=1e-3 -eta_final=1e-4 -eta_threshold=0.90 -batch_size=80 -iterations=1500 -beta=0.01 -keep_probability=0.5 -filename="Commit-test-1" -training_data_aug=True -restart=False -restart_file="test.ckpt"



