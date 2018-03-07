#!/bin/bash
#SBATCH --job-name=coords
#SBATCH --time=24:00:00
#SBATCH --partition=depablo

module load python/2.7-2015q1
bash make_coords_kirk.sh



