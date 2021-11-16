#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00 
#SBATCH --job-name pp_hum
#SBATCH --output pp_human_redo.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
conda activate monocle3
python -u ~/project/scnd/scripts/pp_human_redo.py
