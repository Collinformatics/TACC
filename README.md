# Login To TACC: 

loginTACC.sh allows you to login to a TACC super computer of your choice.

- Define supercomputers with:

      inComputer="vista"

- Define directories with:

      inDirectory="ESM"

After logging in, it will move to a user defined folder in the WORK directory.


Login by with the command: 

    bash loginTACC.sh -u yourUserName

- Flags:

  The following flags will allow you to switch to a secondary server, or directory

  - Login to a secondary computer:

        bash loginTACC.sh -u username -c

  - Start in your secondary directory after logging in:
  
        bash loginTACC.sh -u username -d

# Setting Up A Virtual Environment:

- Create venv:

      python -m venv <venvName>

  - I'm going to name my environment as: venvESM

        python -m venv venvESM

# Install Modules:

- Before you install anything, you need to activate the venv:

      source venvESM/bin/activate

- Its is recommended that you update pip:

      python -m pip install --upgrade pip

- Next, you can install the modules:
  
      pip install fair-esm
      pip install pandas
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# Using TACC:

Move the the WORK directory

    cd /work/file/username

- Tip: use pwd to find your wd

      $ pwd
      /home1/04125/username

  - Now replace "home1" with "work" and change directories with:

        cd /work/04125/username

Activate the venv: (if you have created it)

    source venv/bin/activate

Run a job with sbatch:

    sbatch ESM/runESM.sh

View your jobs:

    squeue -u <username>

Inspect the error logs:

    file runESM*err


