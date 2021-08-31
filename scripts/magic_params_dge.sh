#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=196G
#SBATCH --time=5-00:00:00 
#SBATCH --job-name magicdge
#SBATCH --output magic_params_dge_%J.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate rnavel
python -u magic_params_dge.py
