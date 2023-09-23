#!/bin/bash
# Key parameters
#SBATCH --account=fc_eisenlab
#SBATCH --partition=savio2
#SBATCH --time=48:00:00
#SBATCH --qos=savio_normal
#
# Process parameters
#SBATCH --nodes=10
#SBATCH --cpus-per-task=1
#
# Reporting parameters
#SBATCH --job-name=iqtree_fit
#SBATCH --output=iqtree_fit.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marcsingleton@berkeley.edu
#
# Command(s) to run:
source /global/home/users/singleton/.bashrc
conda activate IDR_evolution
module load gnu-parallel

export WDIR=~/IDR_evolution/analysis/evofit/iqtree_fit/
cd $WDIR

echo $SLURM_JOB_NODELIST | sed s/\,/\\n/g > hostfile

for file in $(ls ../iqtree_meta/out/); do
  if [[ $file == *.afa ]]; then
    echo $(basename $file .afa) $SLURM_CPUS_ON_NODE
  fi
done | parallel --jobs 1 --slf hostfile --wd $WDIR --joblog task.log --resume --colsep ' ' bash iqtree_fit.sh {1} {2}

if [[ -f hostfile ]]; then
  rm hostfile
fi
