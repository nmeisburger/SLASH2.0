#!/bin/bash

#SBATCH --job-name=slash-test
#SBATCH --nodes=4
#SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:02:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all


srun ../build/test/SLASH_test
