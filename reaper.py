import os
import subprocess

input_directory = 'inputs/'
input_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]

output_directory = 'outputs/'
output_files = [f for f in os.listdir(output_directory) if f.endswith('.jpg')]

for input_file in input_files:
    if input_file in output_files:
        continue
    image_file = os.path.join(input_directory, input_file)
    os.remove(image_file)