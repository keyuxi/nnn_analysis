#!/bin/bash
#SBATCH --job-nam=fitSimulatedNUPACK
#SBATCH --output=/scratch/groups/wjg/kyx/NNNlib2b_Nov11/out/%x_%j.out
#SBATCH --error=/scratch/groups/wjg/kyx/NNNlib2b_Nov11/out/%x_%j.err
#SBATCH -n 1
#SBATCH --partition=wjg,biochem,sfgf
#SBATCH --mail-user=kyx@stanford.edu
#SBATCH --mail-type=FAIL,END,START
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate plotting

python3 fit_simulated_NUPACK.py