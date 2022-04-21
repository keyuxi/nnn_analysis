#!/bin/bash
#SBATCH --job-nam=simulateNUPACK
#SBATCH --output=/scratch/groups/wjg/kyx/NNNlib2b_Nov11/out/%x_%j.out
#SBATCH --error=/scratch/groups/wjg/kyx/NNNlib2b_Nov11/out/%x_%j.err
#SBATCH -n 1
#SBATCH --partition=wjg,biochem,sfgf
#SBATCH --mail-user=kyx@stanford.edu
#SBATCH --mail-type=FAIL,END,START
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate nnn

python3 simulate_NUPACK.py