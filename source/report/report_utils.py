import os
import yaml
import ast

import pandas as pd
import matplotlib.pyplot as plt


def lookup_input_output_size(mode):
    if 'cartpole' in mode:
        environment = 'CartPole-v1'
    elif 'acrobot' in mode:
        environment = 'Acrobot-v1'

    if environment == 'CartPole-v1':
        input_size = 4
        output_size = 2
    elif environment == 'Acrobot-v1':
        input_size = 6
        output_size = 3
    else:
        raise ValueError('Unknown environment')

    return input_size, output_size


def preprocess_results(mode):
    path = '../../results'

    path = os.path.join(path, mode)

    episodes = []
    config = {}
    hyperparameters = {}

    n_padding = 20

    for trial_folder in os.listdir(path):

        for run_folder in os.listdir(os.path.join(path, trial_folder)):

            if run_folder == 'config.yml':
                with open(os.path.join(path, trial_folder, run_folder)) as stream:
                    config[trial_folder] = yaml.safe_load(stream)

                continue

            if run_folder == 'hyperparameters.yml':
                with open(os.path.join(path, trial_folder, run_folder)) as stream:
                    hparams = yaml.safe_load(stream)
                    if 'quantum' in mode:
                        hparams['entanglements'] = hparams['entanglements'][0]
                        hparams['rotations'] = '[x, y]'
                    hyperparameters[trial_folder] = hparams

                continue

            csv_path = os.path.join(path, trial_folder, run_folder, 'results.csv')
            df = pd.read_csv(csv_path, sep=';')

            missing_episodes = 100 - df.shape[0]
            if missing_episodes > 0:
                last_n_rows = df.tail(n_padding)
                pad_rows = last_n_rows.sample(missing_episodes, replace=True)
                df = pd.concat([df, pad_rows], ignore_index=True)

            df['trial_id'] = trial_folder
            df['run_id'] = run_folder
            df['environment'] = config[trial_folder]['environment']['name']
            hyperparameter_values = hyperparameters[trial_folder]
            for hyperparameter, value in hyperparameter_values.items():
                df[hyperparameter] = value

            episodes.append(df)

    episodes = pd.concat(episodes, ignore_index=True)
    episodes = episodes.sort_values(by=['trial_id', 'run_id', 'episode'])
    episodes = episodes[[
                            'trial_id',
                            'run_id',
                            'episode',
                            'training_score',
                            'evaluation_score',
                            'epsilon',
                            'training_actions',
                            'evaluation_actions',
                            'evaluation_observations',
                            'evaluation_predictions',
                            'environment'] + list(hyperparameters[next(iter(hyperparameters))].keys())]

    episodes = episodes.dropna(subset=['evaluation_observations', 'evaluation_predictions'])

    # add column number_of_weights
    if 'classic' in mode:
        input_size, output_size = lookup_input_output_size(mode)

        episodes['number_of_weights'] = (input_size * episodes['neurons']) + \
                                        ((episodes['layers']) * episodes['neurons'] ** 2) + \
                                        (episodes['neurons'] * 1)

    return episodes, config


def plot_score(episodes, config, hyperparameters, kind, include_epsilon=False, show_min_and_max=False):
    if kind == 'evaluation':
        score_type = 'evaluation_score'
    elif kind == 'training':
        score_type = 'training_score'
    else:
        raise ValueError('kind must be either evaluation or training')

    score = episodes.groupby(['trial_id', 'episode']).agg(score_mean=(score_type, 'mean'),
                                                          score_max=(score_type, 'max'),
                                                          score_min=(score_type, 'min'),
                                                          epsilon=('epsilon', 'mean'))
    score = score.reset_index()
    score_groups = score.groupby('trial_id')
    fig, ax1 = plt.subplots()
    for name, group in score_groups:
        ax1.plot(group.episode, group.score_mean, label=name)

        if show_min_and_max:
            ax1.plot(group.episode, group.score_max, label=name)
            ax1.plot(group.episode, group.score_min, label=name)

        mode = config[name]['mode']
        print(name, mode, config[name][mode], hyperparameters)

    if include_epsilon:
        ax2 = ax1.twinx()
        for name, group in score_groups:
            ax2.plot(group.episode, group.epsilon, label='epsilon')
            ax2.set_ylabel('Epsilon')

        # ax2.legend()

    environment = config[name]['environment']['name']

    title = f'{kind} score per episode for {environment}'
    if include_epsilon:
        title += ' and epsilon'
    if show_min_and_max:
        title += ' and min/max'
    ax1.set_title(title)
    # ax1.legend()
    plt.show()


def plot_episode(episodes, environment, row=None):
    if not environment:
        print('No environment specified, exiting...')
        raise ValueError('environment must be specified')

    episodes = episodes[episodes['environment'] == environment]

    if row == None:
        row = episodes['evaluation_score'].idxmax()

    predictions = episodes.loc[row, 'evaluation_predictions']
    predictions = ast.literal_eval(predictions)
    observations = episodes.loc[row, 'evaluation_observations']
    observations = ast.literal_eval(observations)
    print(predictions, observations)

    if environment == 'CartPole-v1':
        plot_cartpole_episode(episodes, observations, predictions, row)
    elif environment == 'Acrobot-v1':
        plot_acrobot_episode(episodes, observations, predictions, row)


def plot_cartpole_episode(episodes, observations, predictions, row):
    observations = pd.DataFrame(observations, columns=['position', 'velocity', 'angle', 'angular_velocity'])
    predictions = pd.DataFrame(predictions, columns=['move_left_expected_reward', 'move_right_expected_reward'])
    agent_data = pd.concat([observations, predictions], axis=1)
    agent_data['expected_reward'] = agent_data[['move_left_expected_reward', 'move_right_expected_reward']].max(axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    ax000 = axs[0, 0]
    ax001 = ax000.twinx()
    ax000.set_title('Position')
    ax000.plot(agent_data.index, agent_data['position'], color='blue', label='position')
    ax000.set_ylim(-2, 2)
    ax000.tick_params(axis='y', labelcolor='blue')
    ax000.set_ylabel('position', color='blue')
    ax001.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax001.set_ylim(0, 200)
    ax001.tick_params(axis='y', labelcolor='orange')
    ax001.set_ylabel('expected_reward', color='orange')
    ax010 = axs[0, 1]
    ax011 = ax010.twinx()
    ax010.set_title('Velocity')
    ax010.plot(agent_data.index, agent_data['velocity'], color='blue', label='velocity')
    ax010.set_ylim(-2, 2)
    ax010.tick_params(axis='y', labelcolor='blue')
    ax010.set_ylabel('velocity', color='blue')
    ax011.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax011.set_ylim(0, 200)
    ax011.tick_params(axis='y', labelcolor='orange')
    ax011.set_ylabel('expected_reward', color='orange')
    ax100 = axs[1, 0]
    ax101 = ax100.twinx()
    ax100.set_title('Angle')
    ax100.plot(agent_data.index, agent_data['angle'], color='blue', label='angle')
    ax100.set_ylim(-0.2, 0.2)
    ax100.tick_params(axis='y', labelcolor='blue')
    ax100.set_ylabel('angle', color='blue')
    ax101.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax101.set_ylim(0, 200)
    ax101.tick_params(axis='y', labelcolor='orange')
    ax101.set_ylabel('expected_reward', color='orange')
    ax110 = axs[1, 1]
    ax111 = ax110.twinx()
    ax110.set_title('Angular velocity')
    ax110.plot(agent_data.index, agent_data['angular_velocity'], color='blue', label='angular_velocity')
    ax110.set_ylim(-2, 2)
    ax110.tick_params(axis='y', labelcolor='blue')
    ax110.set_ylabel('angular_velocity', color='blue')
    ax111.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax111.set_ylim(0, 200)
    ax111.tick_params(axis='y', labelcolor='orange')
    ax111.set_ylabel('expected_reward', color='orange')

    fig.suptitle(f'Trial:  {episodes.loc[row, "trial_id"]} \n '
                 f'Episde: {episodes.loc[row, "episode"]} \n'
                 f'Score: {episodes.loc[row, "evaluation_score"]} \n'
                 f'Environment: {episodes.loc[row, "environment"]}')

    fig.text(0.5, 0.04, 'Index', ha='center')
    fig.tight_layout()
    # Create a single legend for all axes
    handles, labels = [], []
    for ax in axs.flatten():
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    plt.show()


def plot_acrobot_episode(episodes, observations, predictions, row):
    observations = pd.DataFrame(observations,
                                columns=['cos_theta1', 'sin_theta1', 'cos_theta2', 'sin_theta2',
                                         'angular_velocity_theta1',
                                         'angular_velocity_theta_2'])
    predictions = pd.DataFrame(predictions, columns=['negative_torque_expected_reward', 'no_torque_expected_reward',
                                                     'positive_torque_expected_reward'])
    agent_data = pd.concat([observations, predictions], axis=1)
    agent_data['expected_reward'] = agent_data[
        ['negative_torque_expected_reward', 'no_torque_expected_reward', 'positive_torque_expected_reward']].max(axis=1)
    fig, axs = plt.subplots(3, 2, figsize=(10, 5))

    ax000 = axs[0, 0]
    ax001 = ax000.twinx()
    ax000.set_title('cos_theta1')
    ax000.plot(agent_data.index, agent_data['cos_theta1'], color='blue', label='cos_theta1')
    ax000.set_ylim(-1, 1)
    ax000.tick_params(axis='y', labelcolor='blue')
    ax000.set_ylabel('cos_theta1', color='blue')
    ax001.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax001.set_ylim(-100, 0)
    ax001.tick_params(axis='y', labelcolor='orange')
    ax001.set_ylabel('expected_reward', color='orange')

    ax010 = axs[1, 0]
    ax011 = ax010.twinx()
    ax010.set_title('sin_theta1')
    ax010.plot(agent_data.index, agent_data['sin_theta1'], color='blue', label='sin_theta1')
    ax010.set_ylim(-1, 1)
    ax010.tick_params(axis='y', labelcolor='blue')
    ax010.set_ylabel('sin_theta1', color='blue')
    ax011.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax011.set_ylim(-100, 0)
    ax011.tick_params(axis='y', labelcolor='orange')
    ax011.set_ylabel('expected_reward', color='orange')

    ax100 = axs[0, 1]
    ax101 = ax100.twinx()
    ax100.set_title('cos_theta2')
    ax100.plot(agent_data.index, agent_data['cos_theta2'], color='blue', label='cos_theta2')
    ax100.set_ylim(-1, 1)
    ax100.tick_params(axis='y', labelcolor='blue')
    ax100.set_ylabel('cos_theta2', color='blue')
    ax101.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax101.set_ylim(-100, 0)
    ax101.tick_params(axis='y', labelcolor='orange')
    ax101.set_ylabel('expected_reward', color='orange')

    ax110 = axs[1, 1]
    ax111 = ax110.twinx()
    ax110.set_title('sin_theta2')
    ax110.plot(agent_data.index, agent_data['sin_theta2'], color='blue', label='sin_theta2')
    ax110.set_ylim(-1, 1)
    ax110.tick_params(axis='y', labelcolor='blue')
    ax110.set_ylabel('sin_theta2', color='blue')
    ax111.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax111.set_ylim(-100, 0)
    ax111.tick_params(axis='y', labelcolor='orange')
    ax111.set_ylabel('expected_reward', color='orange')

    ax120 = axs[2, 0]
    ax121 = ax120.twinx()
    ax120.set_title('angular_velocity_theta1')
    ax120.plot(agent_data.index, agent_data['angular_velocity_theta1'], color='blue', label='angular_velocity_theta1')
    ax120.set_ylim(-2, 2)
    ax120.tick_params(axis='y', labelcolor='blue')
    ax120.set_ylabel('angular_velocity_theta1', color='blue')
    ax121.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax121.set_ylim(-100, 0)
    ax121.tick_params(axis='y', labelcolor='orange')
    ax121.set_ylabel('expected_reward', color='orange')

    ax200 = axs[2, 1]
    ax201 = ax200.twinx()
    ax200.set_title('angular_velocity_theta_2')
    ax200.plot(agent_data.index, agent_data['angular_velocity_theta_2'], color='blue', label='angular_velocity_theta_2')
    ax200.set_ylim(-2, 2)
    ax200.tick_params(axis='y', labelcolor='blue')
    ax200.set_ylabel('angular_velocity_theta_2', color='blue')
    ax201.plot(agent_data.index, agent_data['expected_reward'], color='orange',
               label='expected_reward')
    ax201.set_ylim(-100, 0)
    ax201.tick_params(axis='y', labelcolor='orange')
    ax201.set_ylabel('expected_reward', color='orange')

    plt.suptitle(f'Trial:  {episodes.loc[row, "trial_id"]} \n '
                 f'Episde: {episodes.loc[row, "episode"]} \n'
                 f'Score: {episodes.loc[row, "evaluation_score"]} \n '
                 f'Environment: {episodes.loc[row, "environment"]}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    episodes, description = preprocess_results('quantum_cartpole')
    print(episodes.head())

    # plot_score(episodes, description, 'training', 'CartPole-v1', include_epsilon=True)
    # plot_score(episodes, description, 'evaluation', 'CartPole-v1')
    # plot_score(episodes, description, 'evaluation', 'CartPole-v1', show_min_and_max=True)
    # plot_episode(episodes, 'Acrobot-v1')
    # plot_episode(episodes, 'CartPole-v1')
