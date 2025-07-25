#!/bin/bash

#SBATCH --job-name=ESM
#SBATCH --output=runESM_%j.log
#SBATCH --error=runESM_%j.err
#SBATCH --time=02:00:00            # Set maximum runtime
#SBATCH -N 2                       # Request nodes
#SBATCH -n 10                      # Number of tasks per node (num tasks = -n / -N)
#SBATCH -p gh                      # GPU partition


# | **ESM Model**   | **Param** | **Common Batch Sizes** | **Notes**                                     |
# | --------------- | --------- | ---------------------- | --------------------------------------------- |
# | `esm2_t12_35M`  | 35M       | 64 – 512               | Light model, very fast                        |
# | `esm2_t33_650M` | 650M      | 16 – 64                | Moderate memory use                           |
# | `esm2_t36_3B`   | 3B        | 2 – 8                  | Needs \~20–40 GB GPU                          |
# | `esm2_t48_15B`  | 15B       | 1 – 2                  | Needs \~80–100 GB GPU (e.g., A100 80GB, H100) |

# Input 1: Job Parameters
inModelType='15B Params'
inLayerESM=20
inBatchSize=2

# Input 2: Dataset Parameters
inEnzymeName='Mpro2'
inFixedAA="Q"
inFixedPos="4"
inUseReadingFrame=true
inMinSubs=100 # 5000, 1000, 100, 10
inMinES=0
inLoadSubsForPredictions=false

# Get inputs
while getopts "l:m:ps" opt; do
  case $opt in
    l)
      inLayerESM=$OPTARG
      ;;
    m)
      inMinSubs="$OPTARG"
      ;;
    p)
      inLoadSubsForPredictions=true
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
baseDir=$(pwd)

# Redirect temporary files and cache to work dir
export TORCH_HOME="$baseDir/.cache/hub"
export TMPDIR="$baseDir/tmp"
export TEMP="$baseDir/tmp"
export CACHE_DIR="$baseDir/.cache"
export PYTHON_EGG_CACHE="$baseDir/.python-eggs"

## Retrieve the requested time limit
if [ -z "$SLURM_JOB_ID" ]; then
  echo "Error: SLURM_JOB_ID is not set."
  exit 1
fi

# Set runtime limit
runtimeLimit=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/TimeLimit/ {print $2}' | awk '{print $1}')

# Log the start time
start_time=$(date +%s)

# Log environment and settings
echo -e "\nJob Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Enzyme: $inEnzymeName"
echo "Minimum Substrate Count: $inMinSubs"
echo "ESM Model: $inModelType"
echo "ESM Layer: $inLayerESM"
echo "Total Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo -e "Batch Size: $inBatchSize\n"


# ===============================================================================
# Run your Python script
python ESM/ESM.py "$inModelType" "$inEnzymeName" "$inFixedAA" "$inFixedPos" \
              "$inUseReadingFrame" "$inMinES" "$inMinSubs" "$inBatchSize" \
              "$inLayerESM" "$inLoadSubsForPredictions"

# Log the end time
end_time=$(date +%s)

# Calculate and log the runtime
runtime=$((end_time - start_time))
echo -e "   Total Runtime: $((runtime / 60)):$((runtime % 60)) min\n"
