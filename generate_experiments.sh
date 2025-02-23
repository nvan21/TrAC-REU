#!/bin/bash

bash generate_training_curves.sh
bash generate_noise_curves.sh
bash generate_length_parameter_curves.sh
bash generate_mass_parameter_curves.sh
python3 generate_plots.py

echo "All scripts executed successfully"