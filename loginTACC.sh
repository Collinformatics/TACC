#!/bin/bash

# Define: Computer Cluster
inComputer="vista"

# Define: Project Directory
inDirectory="ESM"


# Get inputs
while getopts "cdu:" opt; do
  case $opt in
    c)
      # Redefine: Computer Cluster
      inComputer="ls6"
      ;;
    d)
      # Redefine: Directory
      inDirectory="EvolvePro"
      ;;
    u)
      userName="$OPTARG"
      ;;
    *)
      exit 1
      ;;
  esac
done

# Check for required flag
if [[ -z "$userName" ]]; then
  echo "ERROR: -u <username> is required"
  echo "Usage: $0 -u <username> [-c] [-d]"
  exit 1
fi


# Display login info
echo "  Computer: $inComputer"
echo -e "  Username: $userName\n"


# Login:
login () {
if [[ $inComputer == "vista" ]]; then
  ssh -l "$userName" login1.vista.tacc.utexas.edu
  exit
else
  ssh -l "$userName" login1.ls6.tacc.utexas.edu
fi
}
login

# Verify if login was successful
dir="$(pwd)"

if [[ ! "$dir" == *"$userName"* ]]; then
  echo "ERROR: Your user name '$userName' is not part of the wd."
  echo "Dir: $dir"
  exit 1
fi


# Move to the working directory
cd \$WORK &&
cd ..


# Make the project directory if it doesn't exist
if [[ ! -d "$inDirectory" ]]; then
  echo "Directory '$inDirectory' does not exist. Creating it now..."
  mkdir -p "$inDirectory" || { echo "ERROR: Could not create directory"; exit 1; }
fi

# Move to project directory
# shellcheck disable=SC2164
cd "$inDirectory" 


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


