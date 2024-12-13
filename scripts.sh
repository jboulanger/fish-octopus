#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --partition=agpu
#SBATCH --gres=gpu:1

echo "TASK ID : $SLURM_ARRAY_TASK_ID"

apptainer exec \
	--writable-tmpfs \
	--bind /cephfs2:/cephfs2,/cephfs:/cephfs,/lmb:/lmb \
	/public/singularity/containers/lightmicroscopy/bioimaging-container/bioimaging.sif \
	/bin/micromamba run -n imaging \
	python octofish.py \
	-s /cephfs/acourtney/HCR_Octopus_Jerome/ClusterTesting \
	-d result \
	-n $SLURM_ARRAY_TASK_ID
