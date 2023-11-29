import os
import subprocess

# Specify the directory containing input files
input_directory = '../inputs/'

# List all files in the input directory
input_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]

# Specify the base command to run
base_command = 'python3 demo.py'

# Iterate over each input file and run the command
for input_file in input_files:
    # Build the full path to the input file
    input_path = os.path.join(input_directory, input_file)

    # Construct the full command
    full_command = f'{base_command} -f {input_path} -o 3d'

    # Run the command using subprocess
    subprocess.run(full_command, shell=True)
