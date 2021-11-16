#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00 
#SBATCH --job-name phateparamscan


module load miniconda
conda activate monocle3

python -u ~/project/scnd/scripts/phate_paramscan.py > ./jobs/hum_phateparamscan_handful.log
mail -s phate_paramscan2 ngravindra@gmail.com <<< "finished yo"

exit