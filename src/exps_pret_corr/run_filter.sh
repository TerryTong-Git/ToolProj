#!/bin/sh
#SBATCH --job-name=filterexps
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH -p p_nlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL
#SBATCH --cpus-per-task=32
#SBATCH --time=1-6:00:00

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models}"
pixi run python filter.py
