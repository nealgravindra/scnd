#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00 
#SBATCH --job-name ftp
#SBATCH --output tape_transfer_%J.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<ngravindra@gmail.com>

bash ~/project/scnd/scripts/transfer_raw_from_tape.sh
