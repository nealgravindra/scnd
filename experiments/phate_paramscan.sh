#!/bin/bash/

module load miniconda
conda activate monocle3
python -u ~/project/scnd/scripts/phate_paramscan.py > ./jobs/hum_phateparamscan.log
mail -s phate_paramscan ngravindra@gmail.com <<< "finished yo"

exit