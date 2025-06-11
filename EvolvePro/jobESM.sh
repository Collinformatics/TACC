#!/bin/bash
#SBATCH --job-name=ESM
#SBATCH --output=jobESM_%j.log
#SBATCH --error=jobESM_%j.err
#SBATCH --time=00:30:00            # Set maximum runtime
#SBATCH --cpus-per-task=4          # Number of CPUs for the task
#SBATCH --nodes=1                  # Request nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (usually 1 per node)
#SBATCH --mem=90G                 # Memory
#SBATCH --partition=gpu-h100       # GPU partition


# Define: Enzyme name
enzymeName="Mpro2"

# Define: File paths
pathFasta=$(pwd)/output${enzymeName}/variants${enzymeName}.fasta
pathSave=$(pwd)/output/plm/esm/${enzymeName}
pathCatDir=$(pwd)/output/plm/esm/

# Define: Batch size
batch=512


# ===============================================================================

# Redirect temporary files and cache to /work
export TORCH_HOME=/work/07687/collin25/.cache/hub
export TMPDIR=/work/07687/collin25/tmp
export TEMP=/work/07687/collin25/tmp
export CACHE_DIR=/work/07687/collin25/.cache
export PYTHON_EGG_CACHE=/work/07687/collin25/.python-eggs

# Redirect temporary files and cache to /work
export TORCH_HOME=/work/07687/collin25/.cache/hub
export TMPDIR=/work/07687/collin25/tmp
export TEMP=/work/07687/collin25/tmp
export CACHE_DIR=/work/07687/collin25/.cache
export PYTHON_EGG_CACHE=/work/07687/collin25/.python-eggs

# Retrieve the requested time limit
requested_time=$(scontrol show job $SLURM_JOB_ID | awk -F= '/TimeLimit/ {print $2}' | awk '{print $$


# Log the start time
start_time=$(date +%s)

# Log environment and settings
echo -e "\nJob Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Enzyme: $enzymeName"
echo "Total Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)"
echo "Node List: $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory Allocation: ${SLURM_MEM_PER_NODE}"
echo "Partition: $SLURM_JOB_PARTITION"
echo -e "Batch Size: $batch\n"

# Run your Python script
python3 EvolvePro/plm/esm/extract.py esm2_t48_15B_UR50D \
  "$pathFasta" \
  "$pathSave" \
  --toks_per_batch $batch \
  --include mean \
  --concatenate_dir "$pathCatDir"


# Log the end time
end_time=$(date +%s)

# Calculate and log the runtime
runtime=$((end_time - start_time))
echo -e "\nRequested Time: $requested_time"
echo -e "   ESM Runtime: $((runtime / 60)):$((runtime % 60))\n"
