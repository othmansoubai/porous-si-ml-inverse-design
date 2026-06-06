#!/bin/bash
#SBATCH --job-name=nemd_porous
#SBATCH --partition=mediumq
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos

PAXLO=0;  PAXHI=4;  PAYLO=0;  PAYHI=4
PBXLO=2;  PBXHI=6;  PBYLO=2;  PBYHI=6
SEED=12345
LABEL="example_S4_quarter"

mpirun --mca pml ob1 --mca btl ^vader,openib \
  lmp -in in.nemd_porous_parametric \
  -var pAxlo $PAXLO -var pAxhi $PAXHI -var pAylo $PAYLO -var pAyhi $PAYHI \
  -var pBxlo $PBXLO -var pBxhi $PBXHI -var pBylo $PBYLO -var pByhi $PBYHI \
  -var seed $SEED -var label $LABEL

mkdir -p results
mv temp_profile_${LABEL}_seed${SEED}.dat results/ 2>/dev/null
mv e_transfer_${LABEL}_seed${SEED}.dat   results/ 2>/dev/null

echo "Done: $LABEL seed $SEED"
