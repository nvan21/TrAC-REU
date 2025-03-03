# Investigating Model-Free vs Model-Based RL for Sim-to-Real Transfer

This repository contains an implementation of the [Short Horizon Actor Critic](https://arxiv.org/pdf/2204.07137) (SHAC) algorithm, along with benchmarks against popular model-free algorithms (PPO and SAC) on the Pendulum environment from OpenAI Gymnasium.

## Project Overview

This research was conducted as part of the [TrAC REU (Research Experience for Undergraduates) program](https://trac-ai.iastate.edu/education/reu/) during Summer 2024. The main focus was investigating the effectiveness of model-free versus model-based reinforcement learning algorithms in addressing the simulation-to-real gap.

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

## Research Poster

![Iowa State Summer Undergraduate 2024 Poster Presentation](assets/REU_presentation.png)

## Key Findings

My experiments demonstrated several interesting and somewhat surprising results:

- **SAC outperformed both PPO and SHAC** across all tests
- SAC converged to expert-level performance with **5x fewer timesteps** than the other methods
- SAC policies showed **greater robustness** to observation noise and system parameter tweaks
- SHAC can obtain expert performance but sometimes fails with certain seeds, likely due to getting stuck in local minima
- Model-free algorithms (SAC, PPO) proved to be more robust and efficient than the model-based approach (SHAC)
- These results actually contradict the findings in the original SHAC paper, possibly due to optimizations in their simulator that were not implemented in my environment
- While SAC demonstrated superior performance, this came at the cost of increased training time relative to PPO and SHAC

## Training Setup

All experiments were run with the following hardware configuration:

- **GPU**: NVIDIA A100
- **CUDA**: Version 11.8
- **CPU**: 8 cores of an AMD EPYC 8654
- **RAM**: 32 GB

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
cd TrAC-REU

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

To reproduce the experiments, run:

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

The findings in this repository represent preliminary results that will inform later experiments. Future work will include:

1. **Double Pendulum Environment**: Testing these algorithms on the more complex double pendulum system
2. **Physical Implementation**: Developing a physical system of the double pendulum environment
3. **Policy Transfer Testing**: Evaluating how well policies trained in simulation transfer to the real physical system
4. **Sim-to-Real Bridge Metrics**: Creating better representations of how well each algorithm bridges the simulation-to-real gap

## Acknowledgments

Thanks to Prajwal Koirala for his helpful feedback and ideas, Cody Fleming for his guidance and support, and to the TrAC REU (funded by the AI Institute for Resilient Agriculture - AIIRA) for making this research possible.
