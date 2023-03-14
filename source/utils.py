import os
import random

import yaml
import numpy as np

from source.TrainingHandler import path, config


def load_yml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def reverse_one_hot(value, length):
    output = np.zeros(length)
    output[value] = 1.0
    return output


def generate_bounds(bounds):
    for key in bounds.keys():
        if key == 'rotations' or key == 'entanglements':
            bounds[key] = [bounds[key] for _ in range(bounds['layers'])]

    return bounds


def save_results_to_file(created_at, run, episode, training_score, eval_score, epsilon, training_actions, eval_action,
                         eval_observations, eval_predictions):
    with open(f'{path}results/{created_at}/run_{run}/results.csv', 'r+') as f:
        epsilon = round(epsilon, 2)
        eval_observations = [list(x) for x in eval_observations]
        eval_predictions = [list(x[0]) for x in eval_predictions]
        f.seek(0, os.SEEK_END)
        f.write(
            f'{episode};{training_score};{eval_score};{epsilon};{training_actions};{eval_action};{eval_observations};{eval_predictions}\n')


class Hyperparameters:
    def __init__(self):
        mode = config['mode']
        self.bounds = config[mode]['bounds']
        self.all_configs = self.generate_all_configs()

        self.used_configs = []

    def generate_all_configs(self):
        all_configs = []
        keys = list(self.bounds.keys())

        def _helper(idx, config):
            if idx == len(keys):
                all_configs.append(config)
            else:
                key = keys[idx]
                vals = self.bounds[key]
                for val in vals:
                    new_config = config.copy()
                    new_config[key] = val
                    _helper(idx + 1, new_config)

        _helper(0, {})
        return all_configs

    def select_hyperparameters(self):
        if not self.all_configs:
            raise ValueError("All possible configurations have been used")

        new_config = random.choice(self.all_configs)
        self.all_configs.remove(new_config)
        self.used_configs.append(new_config)
        return new_config
