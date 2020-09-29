#!/bin/bash

#SBATCH --job-name="CTaws2"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --workdir=/data/scratch/torresc/cosipy_old/utilities/aws2cosipy/
#SBATCH --account=gaby-vasa
#SBATCH --error=/data/scratch/torresc/cosipy_old/utilities/aws2cosipy/control2cosipy.err
#SBATCH --output=/data/scratch/torresc/cosipy_old/utilities/aws2cosipy/control2cosipy.out

echo $SLURM_CPUS_ON_NODE

export PATH="/nfsdata/programs/anaconda3_201812/bin:$PATH"
python -u /data/scratch/torresc/cosipy_old/utilities/aws2cosipy/aws2cosipy.py -c ../../data/input/Peru/data_aws_peru.csv -o ../../data/input/Peru/Peru_input_8.nc -s ../../data/static/Peru_static_50m.nc -b 20160901 -e 20170831
