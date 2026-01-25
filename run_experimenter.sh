#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --array=1-50
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J priorbai
#SBATCH -p normal
#SBATCH -A hpc-prf-intexml
#SBATCH --mail-user m.wever@ai.uni-hannover.de
#SBATCH --mail-type ALL

cd /scratch/hpc-prf-intexml/wever/priorbai/PriorBAI/

source .venv/bin/activate

python -m priorbai.priorbai
