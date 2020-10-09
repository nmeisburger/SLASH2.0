#!/bin/bash

#SBATCH --job-name=slash-run
#SBATCH --nodes=8
#SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:02:00
#SBATCH --mail-user=jz77@rice.edu
#SBATCH --mail-type=all


srun ../build/src/SLASH_run