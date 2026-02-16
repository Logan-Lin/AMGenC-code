#!/bin/bash
#SBATCH --output=stdout/%x.out
#SBATCH --time=6-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=prioritized
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -e

singularity run --nv runtime/charge-bal.sif "$@"
