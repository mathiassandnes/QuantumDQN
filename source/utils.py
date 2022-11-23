import yaml
import numpy as np

def load_yml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def reverse_one_hot(value, length):
    output = np.zeros(length)
    output[value] = 1.0
    return output
