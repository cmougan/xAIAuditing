#!/bin/bash
 
#SBATCH --job-name=job_name      # nom du job
#SBATCH --time=06:00:00            # Temps d�~@~Yexécution maximum demande (HH:MM:SS)
#SBATCH --output=output/output%x%j.out  # Nom du fichier de sortie
#SBATCH --error=error/error%x%j.out   # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --nodes=1
#SBATCH --ntasks=1                # Nombre total de processus MPI
#SBATCH --mem=64G
 
 
 
module load conda/py3-latest
 
source activate py310

make run_all_datasets