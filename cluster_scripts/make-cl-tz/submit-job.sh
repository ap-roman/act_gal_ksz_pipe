#!/bin/bash
#SBATCH -A aroman
#SBATCH -t 0-6:00              # time limit: (D-HH:MM) 

# sbatch --partition=debugq --ntasks=4 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh
echo "passing config file $1";
# sbatch --partition=amdq --ntasks=12 -t 0-16:00 --cpus-per-task=32 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh $1;
sbatch --partition=defq --ntasks=16 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh $1;
# sbatch --partition=debugq --ntasks=4 --cpus-per-task=20 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh $1;
# sbatch --partition=amdq --ntasks=16 -t 0-6:00 --cpus-per-task=8 --mail-type=ALL --mail-user=aroman@perimeterinstitute.ca slurm-script.sh