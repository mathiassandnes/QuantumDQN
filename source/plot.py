import os
import yaml
import matplotlib.pyplot as plt

# Set the path to the results folder
path = '../results'

# Iterate over all subfolders in the results folder
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)

    # Check if the folder contains results
    if os.path.isfile(os.path.join(folder_path, 'results.txt')):
        # Load the data
        with open(os.path.join(folder_path, 'results.txt'), 'r') as f:
            data = f.read()
            data = data[1:]
            data = data.split(',')
            # remove empty string
            data = [d for d in data if d != '']
            data = [float(d) for d in data]

        # Load the config file
        with open(os.path.join(folder_path, 'config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        config_without_bounds = config.copy()
        config_without_bounds['quantum']['bounds'] = None

        print(folder_path + ' ' + str(config_without_bounds['quantum']))

        # Plot the data
        plt.plot(data, label=folder)

# need function for this that consideres environment
plt.ylim(0, 100)
plt.legend()
plt.show()
