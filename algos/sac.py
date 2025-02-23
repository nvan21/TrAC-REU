import stable_baselines3
from utils.hyperparameters import Experiments


class SAC(stable_baselines3.SAC):
    def __init__(self):
        super().__init__()

    def run_mass_experiment(self):
        for id, experiment in Experiments.experiments.values():
            pass

    def run_length_experiment(self):
        pass
