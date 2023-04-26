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

        try:
            with open(os.path.join(subfolder_path, 'config.yml'), 'r') as config_file:
                config_data = yaml.safe_load(config_file)
        except FileNotFoundError:
            continue

        # Extract the required properties
        mode = config_data['mode']
        environment_name = config_data['environment']['name']

        # Check if the trainable rotations flag should be added
        if 'quantum' in config_data and 'trainable_rotations' in config_data['quantum'] and config_data['quantum']['trainable_rotations']:
            environment_name += '_TR'

        # Create a new folder name
        new_folder_name = f"{mode}{environment_name}"
        new_folder_path = os.path.join(base_folder, new_folder_name)

        # Create the new folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Move the subfolder into the new folder
        os.rename(subfolder_path, os.path.join(new_folder_path, subfolder))


# iterate over all subfolders in the base folder
for subfolder in os.listdir(base_folder):
    # if filename contains the string "run" and contains a folder named "models" which is empty
    if "run" in subfolder and os.path.isdir(os.path.join(base_folder, subfolder, "models")) and not os.listdir(os.path.join(base_folder, subfolder, "models")):
        # delete the folder
        os.rmdir(os.path.join(base_folder, subfolder, "models"))
