#!/bin/bash

# Define: Computer Cluster
inComputer="vista"

# Define: Project Directory
inDirectory="ESM"


# Get inputs
while getopts "elu:" opt; do
  case $opt in
    e)
      inDirectory="EvolvePro"
      ;;
    l)
      # Redefine: Computer Cluster
      inComputer="ls6"
      ;;
    u)
      userName="$OPTARG"
      ;;
    *)
      exit 1
      ;;
  esac
done

# Display login info
echo "Computer: $inComputer"
echo -e "Username: $userName\n"


# Login:
login () {
if [[ $inComputer == "vista" ]]; then
  ssh -l "$userName" login1.vista.tacc.utexas.edu
else
  ssh -l "$userName" login1.ls6.tacc.utexas.edu
fi
}


# Move to the working directory
cd \$WORK &&
cd ..

# Make the directory if it doesn't exist
if [[ ! -d "$inDirectory" ]]; then
  echo "Directory '$inDirectory' does not exist. Creating it now..."
  mkdir -p "$inDirectory" || { echo "ERROR: Could not create directory"; exit 1; }
fi


# Move to project directory
#cd "$inDirectory"
dir=$(pwd)
echo "Directory: $dir"


# Inspect: Virtual Environment
if [[ ! -d "venv" ]]; then
  echo "Virtual environment 'venv' was not found in: $inDirectory"
  echo -e "     Creating a new environment: venv\n"

  # Create venv
  python3 -m venv venv
  chmod +x venv/bin/activate

  # Activate virtual environment
  source venv/bin/activate

  # Install python modules
  pip install --upgrade pip
  pip install cupy-cuda12x
  pip install fair-esm
  pip install matplotlib
  pip install numpy
  pip install pandas
  pip install seaborn
  pip install scikit-learn
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install xgboost
fi
