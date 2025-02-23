#!/bin/bash

# Collect the SHAC trajectories
python3 generate_training_curves_shac.py

# Check if the first script executed successfully
if [ $? -ne 0 ]; then
  echo "shac failed"
  exit 1
fi

# Collect the PPO trajectories
python3 generate_training_curves_ppo.py

# Check if the second script executed successfully
if [ $? -ne 0 ]; then
  echo "ppo failed"
  exit 1
fi

# Collect the SAC trajectories
python3 generate_training_curves_sac.py

# Check if the third script executed successfully
if [ $? -ne 0 ]; then
  echo "sac failed"
  exit 1
fi

# Create the plot
python3 generate_training_curves_plots.py

# Check if the fourth script executed successfully
if [ $? -ne 0]; then
  echo "plot generation failed"
  exit1
fi

echo "All scripts executed successfully"