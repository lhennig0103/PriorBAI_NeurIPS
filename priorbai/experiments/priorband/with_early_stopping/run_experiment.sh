#!/bin/bash
#SBATCH -t 0:20:00
#SBATCH --array=1-10
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J priorband_es
#SBATCH -p normal
#SBATCH --mail-user l.fehring@ai.uni-hannover.de
#SBATCH --mail-type ALL
#SBATCH -o logs/priorband/with_early_stopping/%A_%a.out
#SBATCH -e logs/priorband/with_early_stopping/%A_%a.err

cd /scratch/hpc-prf-intexml/fehring/PriorBAI/

source .venv/bin/activate

python priorbai/experiments/priorband/with_early_stopping/run.py
