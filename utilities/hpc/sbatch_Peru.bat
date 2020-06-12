#!/bin/bash

#SBATCH --job-name="PeMaNo"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --workdir=/data/scratch/torresc/cosipy_peru/
#SBATCH --account=gaby-vasa
#SBATCH --error=/data/scratch/torresc/cosipy_peru/Control_master.err
#SBATCH --partition=computehm
#SBATCH --output=/data/scratch/torresc/cosipy_peru/Control_master.out

echo $SLURM_CPUS_ON_NODE

export PATH="/nfsdata/programs/anaconda3_201812/bin:$PATH"
python -u /data/scratch/torresc/cosipy_peru/COSIPY.py
