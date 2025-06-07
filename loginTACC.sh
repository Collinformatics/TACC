#!/bin/bash

# Connect to lonestar6 with forced pseudo-terminal allocation
ssh -t collin25@login1.ls6.tacc.utexas.edu "
    # Move to the working directory
    cd \$WORK &&
    cd ..
    #echo \"WD: \$(pwd)\"

    # Check if the virtual environment exists; if not, create and activate it
    if [ ! -d \"venv\" ]; then
        echo \"Virtual environment not found. Creating a new one...\"
        python3 -m venv venv
        chmod +x venv/bin/activate
    fi


    # Activate virtual environment
    source venv/bin/activate &&    
    
    # Change directory
    cd ls6/EvolvePro 

    # Start interactive shell to keep the session alive
    bash --login
"

