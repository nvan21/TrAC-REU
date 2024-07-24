#!/bin/bash

python3 generate_training_curves_plots.py
python3 generate_noise_curves_plots.py
python3 generate_mass_parameter_curves_plots.py
python3 generate_length_parameter_curves_plots.py

echo "All scripts executed successfully"