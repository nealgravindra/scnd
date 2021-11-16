#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=121G
#SBATCH --time=5-00:00:00 
#SBATCH --job-name hum_dge
#SBATCH --output hum_dge_%J.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
source activate py385dev
python -u hum_dge_sca3.py
