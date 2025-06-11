#!/bin/bash
#!/bin/bash
#SBATCH --job-name=ESM
#SBATCH --output=runESM_%j.log
#SBATCH --error=runESM_%j.err
#SBATCH --time=00:30:00            # Set maximum runtime
#SBATCH --cpus-per-task=4          # Number of CPUs for the task
#SBATCH --nodes=1                  # Request nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (usually 1 per node)
#SBATCH --partition=gpu-h200       # GPU partition
#SBATCH --mem=128G                 # Memory {15B: 128G, 3B: 80G}


# | **ESM Model**   | **Param** | **Common Batch Sizes** | **Notes**                                     |
# | --------------- | --------- | ---------------------- | --------------------------------------------- |
# | `esm2_t12_35M`  | 35M       | 64 – 512               | Light model, very fast                        |
# | `esm2_t33_650M` | 650M      | 16 – 64                | Moderate memory use                           |
# | `esm2_t36_3B`   | 3B        | 2 – 8                  | Needs \~20–40 GB GPU                          |
# | `esm2_t48_15B`  | 15B       | 1 – 2                  | Needs \~80–100 GB GPU (e.g., A100 80GB, H100) |


inModelType='15B Params'
inEnzymeName='Mpro2'
inSubstrateLength=8
inUseReadingFrame=true
inMinSubs=100 # 5000, 1000, 100, 10
inMinES=0
batchSize=2
AA="Q"
pos="4"
fileNameSubsPred=false

# Get inputs
while getopts "m:ps" opt; do
  case $opt in
    m)
      inMinSubs="$OPTARG"
      ;;
    p)
      fileNameSubsPred=true
      ;;
    s)
      inModelType='3B Params'
      ;;
    *)
      exit 1
      ;;
  esac
done


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

# ===============================================================================

# Run your Python script
python ESM.py "$inModelType" "$inEnzymeName" "$AA" "$pos" "$inSubstrateLength" \
              "$inUseReadingFrame" $inMinES "$inMinSubs" "$batchSize" "$fileNameSubsPred"


# Log the end time
end_time=$(date +%s)

# Calculate and log the runtime
runtime=$((end_time - start_time))
echo -e "\nRequested Time: $requested_time"
echo -e "   Total Runtime: $((runtime / 60)):$((runtime % 60))\n"
