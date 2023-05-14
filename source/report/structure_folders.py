import os
import shutil
import yaml

# Set the base folder
base_folder = '../../results'

# Iterate over subfolders in the base folder
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):

        # Delete empty runs and trials
        trial_folders = [folder for folder in os.listdir(subfolder_path) if folder.startswith('run_')]
        for trial_folder in trial_folders:
            trial_folder_path = os.path.join(subfolder_path, trial_folder)
            models_folder_path = os.path.join(trial_folder_path, 'models')

            # Check if the models folder is empty
            if os.path.exists(models_folder_path) and not os.listdir(models_folder_path):
                shutil.rmtree(trial_folder_path)

        # If there are no runs left in the trial, delete the trial
        remaining_trial_folders = [folder for folder in os.listdir(subfolder_path) if folder.startswith('run_')]
        if not remaining_trial_folders:
            shutil.rmtree(subfolder_path)
            continue

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
        if config_data['quantum'].get('trainable_entanglements') is True and mode == 'quantum':
            environment_name += '_TR'

        # Create a new folder name
        new_folder_name = f"{mode}{environment_name}"
        new_folder_path = os.path.join(base_folder, new_folder_name)

        # Create the new folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Move the subfolder into the new folder
        os.rename(subfolder_path, os.path.join(new_folder_path, subfolder))
