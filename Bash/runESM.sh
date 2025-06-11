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


inModelType='3B Params'
inEnzymeName='Mpro2'
inUseReadingFrame=true
inMinSubs=100

# Get inputs
while getopts "b:r:p:l" opt; do
  case $opt in
    b)
      batchSize=$OPTARG
      ;;
    r)
      AA=$OPTARG
      ;;
    p)
      pos=$OPTARG
      ;;
    l)
      inModelType='15B Params'
      ;;
    *)
      exit 1
      ;;
  esac
done

if [[ $inModelType == "15B Params" ]]; then
  #SBATCH --mem=128G                  # Memory
  echo "128 GB"
else
  #SBATCH --mem=80G                  # Memory
  echo "80 GB"
fi

python ESM.py "$inModelType" "$inEnzymeName" "$AA" "$pos" "$inUseReadingFrame" "$inMinSubs" "$batchSize"
