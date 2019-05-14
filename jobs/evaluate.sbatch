#!/bin/bash

#BATCH --job-name=3daudio-ass-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=62GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output=3daudio-ass_%j.out
#SBATCH --err=3daudio-ass_%j.err

module purge
source activate birdvox
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

cd /home/jtc440/dev/3daudio/ambisonic_source_separation/src
python evaluate.py /scratch/jtc440/ambisonic_source_separation/testing/model.h5 /beegfs/jtc440/3daudio/datasets_split/test.txt /beegfs/jtc440/3daudio/datasets/noise/freesound_bformat /beegfs/jtc440/3daudio/srir/isophonics /scratch/jtc440/ambisonic_source_separation/testing
