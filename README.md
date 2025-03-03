# SHAC: Investigating Model-Free vs Model-Based RL for Sim-to-Real Transfer

This repository contains an implementation of the [Short Horizon Actor Critic](https://arxiv.org/pdf/2204.07137) (SHAC) algorithm, along with benchmarks against popular model-free algorithms (PPO and SAC) on the Pendulum-v1 environment from OpenAI Gymnasium.

## Project Overview

This research was conducted as part of the [TrAC REU (Research Experience for Undergraduates) program]() during Summer 2024. The main focus was investigating the effectiveness of model-free versus model-based reinforcement learning algorithms in addressing the simulation-to-real gap.

The sim-to-real gap refers to the performance difference when transferring policies trained in simulation to real-world environments. This gap often occurs due to modeling inaccuracies, parameter uncertainty, and sensor noise that exist in real-world scenarios but may be absent or idealized in simulations.

## Key Research Questions

1. How do model-based methods like SHAC compare to model-free methods like PPO and SAC in handling environmental parameter variations?
2. What is the impact of sensor noise on policy performance?
3. How do these algorithms respond to variations in physical parameters (mass, length) of the pendulum?

## Repository Structure

- `algos/`: Implementation of RL algorithms (SHAC, PPO, SAC)
- `network/`: Neural network architectures for policies and value functions
- `utils/`: Environment wrappers, hyperparameter configurations, and utility functions
- `scripts/`: Scripts for generating training and evaluation data
- `*.sh`: Shell scripts for running experiment batches

## Experiments

The project compares algorithms across multiple dimensions:

1. **Training Performance**: Basic learning curves showing reward vs. timesteps
2. **Mass Parameter Robustness**: Testing how policies perform when pendulum mass varies
3. **Length Parameter Robustness**: Testing how policies perform when pendulum length varies
4. **Noise Robustness**: Testing policy resilience to observation noise

## Key Findings

(Note: This section should be filled with your actual research findings)

Our experiments demonstrated that:

- SHAC shows improved robustness to parameter variations compared to model-free methods
- The short-horizon planning approach helps mitigate the sim-to-real gap
- Observation noise affects model-free methods more severely than model-based approaches

## Installation and Usage

### Prerequisites

- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib
- Wandb (optional, for logging)

### Setup

```bash
# Clone the repository
git clone https://github.com/nvan21/TrAC-REU.git
cd SHAC

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

To reproduce our experiments, run:

```bash
# Run all experiments
bash generate_experiments.sh

# Or run specific experiment types
bash generate_training_curves.sh
bash generate_mass_parameter_curves.sh
bash generate_length_parameter_curves.sh
bash generate_noise_curves.sh

# Generate plots from experimental data
python generate_plots.py
```

## Future Work

The current implementation focuses on the Pendulum-v1 environment for easier debugging and analysis. Future work will port this to the F1TENTH simulator to investigate sim-to-real transfer in more complex robotic systems.

## Acknowledgments

This research was supported by the [TrAC REU program]() during Summer 2024. We thank the program organizers and mentors for their guidance and support throughout this project.

## Citation

If you use this code in your research, please cite:

```
@misc{shac2024,
  author = {Your Name},
  title = {Investigating Model-Free vs Model-Based RL for Sim-to-Real Transfer},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/SHAC}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
