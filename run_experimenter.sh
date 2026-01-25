#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --array=1-300
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu 4G
#SBATCH -J priorbai
#SBATCH -p normal
#SBATCH --mail-user l.fehringr@ai.uni-hannover.de
#SBATCH --mail-type ALL
#SBATCH -o logs/priorbai_%A_%a.out
#SBATCH -e logs/priorbai_%A_%a.err

cd /scratch/hpc-prf-intexml/fehring/PriorBAI/

source .venv/bin/activate

python priorbai/priorbai.py
