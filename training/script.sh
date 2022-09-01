#!/usr/bin/bash

#SBATCH -J "biobert_gmw_FT"   # job name
#SBATCH --mail-type=END  # send at the end
#SBATCH --time=20:00:00   # walltime
#SBATCH --ntasks=24   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1  # extra general resources (one gpu here)

python nlp_pipeline_translation.py