#!/bin/bash
#SBATCH -t 0:20:00
#SBATCH --array=1-680
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J priorband_no_es
#SBATCH -p normal
#SBATCH --mail-user l.fehring@ai.uni-hannover.de
#SBATCH --mail-type ALL
#SBATCH -o logs/priorband/no_early_stopping/%A_%a.out
#SBATCH -e logs/priorband/no_early_stopping/%A_%a.err

cd /scratch/hpc-prf-intexml/fehring/PriorBAI/

source .venv/bin/activate

python priorbai/experiments/priorband/no_early_stopping/run.py
