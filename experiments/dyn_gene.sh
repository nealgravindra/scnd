#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00 
#SBATCH --job-name dyn_gene

module load miniconda
conda activate monocle3

python -u ~/project/scnd/scripts/dynamical_genes_revision.py > ./jobs/dynamical_genes_revision.log
mail -s dynamical_genes_revision ngravindra@gmail.com <<< "finished yo"

exit