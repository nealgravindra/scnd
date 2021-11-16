#!/bin/bash

#SBATCH --partition=general
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

module load cellranger/6.0.1
cd $3
cellranger count --id=$1 --sample=$1 --fastqs=$4 --transcriptome=$2 --expect-cells=10000
printf '\n... cell counts matrix made for %s\n\nexiting.' $1
