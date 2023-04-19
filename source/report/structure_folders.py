import os
import yaml

# Set the base folder
base_folder = '../../results'

# Iterate over subfolders in the base folder
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Read the config.yml file
        with open(os.path.join(subfolder_path, 'config.yml'), 'r') as config_file:
            config_data = yaml.safe_load(config_file)

        # Extract the required properties
        mode = config_data['mode']
        environment_name = config_data['environment']['name']

        # Create a new folder name
        new_folder_name = f"{mode}_{environment_name}"
        new_folder_path = os.path.join(base_folder, new_folder_name)

        # Create the new folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Move the subfolder into the new folder
        os.rename(subfolder_path, os.path.join(new_folder_path, subfolder))
