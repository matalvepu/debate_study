#!/bin/sh


#SBATCH -p gpu --gres=gpu -C K80
#SBATCH --mem=20gb
#SBATCH -c 4
#SBATCH -a 0-2
#SBATCH -t 0-23:00:00  
#SBATCH -J albert_mhasan8
#SBATCH -o /scratch/mhasan8/output/albert_output%j
#SBATCH -e /scratch/mhasan8/output/albert_error%j
#SBATCH --mail-type=all    

module load anaconda3/5.3.0b
source activate /home/mhasan8/.conda/envs/pydeep
module load git


wandb agent mhasan/debate/2i0rk1fj