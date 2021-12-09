#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=119G
#SBATCH --time=4-00:00:00 
#SBATCH --job-name dyn_gene
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load miniconda
conda activate monocle3

python -u ~/project/scnd/scripts/dynamical_genes_revision.py 2>&1 > ./jobs/dynamical_genes_revision.log
mail -s dynamical_genes_revision ngravindra@gmail.com <<< "finished yo"

exit
