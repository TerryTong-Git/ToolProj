#SBATCH --job-name=filterexps
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH -p p_nlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --cpus-per-task=32
#SBATCH --time=1-6:00:00

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="/nlpgpu/data/terry/ToolProj/src/models"
pixi run python filter.py