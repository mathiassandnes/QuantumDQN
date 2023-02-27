import numpy as np
import matplotlib.pyplot as plt
import yaml

path = '../results/10-02-2023_16-01-20'

# Load the data
with open(path + '/results.txt', 'r') as f:
    data = f.read()
    data = data[1:]
    data = data.split(',')
    data = [float(d) for d in data]

# load yaml as dict
with open(path + '/config.yml', 'r') as f:
    config = yaml.safe_load(f)
    print(config['quantum'])

    print(config)



# plot the data

plt.plot(data)
plt.show()
