#!/bin/bash
#SBATCH --job-name=learning
#SBATCH --time=36:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1

module load singularity
/software/src/singularity_images/tensorflow_1.1.0.img 367.44 3 learning_rate_repetitions.py -eta=1e-3 -batch_size=80 -epochs=1200 -beta=0.01 -keep_probability=0.5 



