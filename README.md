# Login To TACC: 

loginTACC.sh allows you to login to a TACC super computer of your choice.

- Define supercomputers with:

      inComputer="vista"

- Define directories with:

      inDirectory="ESM"

After logging in, it will move to a user defined folder in the WORK directory.


Run the with the command: 

    bash loginTACC.sh -u yourUserName

- Flags:

  The following flags will allow you to switch to a secondary server, or directory

  - Start in your secondary directory after logging in:
  
        bash loginTACC.sh -u username -e

  - Login to a secondary computer:

        bash loginTACC.sh -u username -l
   
