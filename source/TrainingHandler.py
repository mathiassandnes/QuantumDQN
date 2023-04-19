import os
import numpy as np

import yaml

from source.Agent import Agent
from source.gym_handler import GymHandler
from source.utils import load_yml, generate_bounds, save_results_to_file

config = load_yml(f'../configuration.yml')
path = '' if config['cluster'] else '../'


def run_evaluation_episodes(environment, agent, render=False):
    total_rewards = []
    all_actions = []
    all_observations = []
    all_predictions = []

    for episode in range(config['training']['eval_episodes']):
        observation = environment.reset()
        observation = observation[0]
        total_reward = 0
        actions = []
        observations = []
        predictions = []

        for step in range(config['training']['max_eval_steps']):
            if config['environment']['render'] or render:
                environment.render()

            action, prediction = agent.choose_action(observation, training=False, include_raw=True)

            actions.append(action)
            observations.append(observation)
            predictions.append(prediction)

            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if config['verbose']:
            print(actions)

        total_rewards.append(total_reward)
        all_actions.append(actions)
        all_observations.append(observations)
        all_predictions.append(predictions)

    print(f'eval: {total_rewards}')

    return total_rewards, all_actions, all_observations, all_predictions


def run_training_episode(environment, agent, observation):
    actions = []
    timestep = 0
    for timestep in range(config['training']['max_steps']):
        action = agent.choose_action(observation, training=True)
        actions.append(action)

        next_observation, reward, terminated, truncated = environment.step(action)

        agent.remember(observation, action, reward, next_observation, terminated)
        agent.learn()

        observation = next_observation

        if config['environment']['render']:
            environment.render()

        if terminated or truncated:
            break

    return timestep, actions


class TrainingHandler:
    def __init__(self, run, created_at, hyperparameters):
        self.hyperparameters = generate_bounds(hyperparameters)
        self.run_num = run

        self.created_at = created_at
        self.init_result_files()

        self.environment = GymHandler()
        observation_space = self.environment.env.observation_space
        action_space = self.environment.env.action_space

        self.agent = Agent(observation_space, action_space, hyperparameters=hyperparameters)

    def init_result_files(self):
        save_path = f'{path}results/{self.created_at}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if not os.path.exists(f'{path}results/{self.created_at}/config.yml'):
            with open(f'{path}results/{self.created_at}/config.yml', 'w+') as f:
                yaml.dump(config, f, default_flow_style=False)

        if not os.path.exists(f'{path}results/{self.created_at}/hyperparameters.yml'):
            with open(f'{path}results/{self.created_at}/hyperparameters.yml', 'w+') as f:
                yaml.dump(self.hyperparameters, f, default_flow_style=False)

        os.mkdir(f'{save_path}/run_{self.run_num}')
        os.mkdir(f'{save_path}/run_{self.run_num}/models')

        with open(f'{save_path}/run_{self.run_num}/results.csv', 'w+') as f:
            f.write(
                'episode;training_score;evaluation_score;epsilon;training_actions;evaluation_actions;evaluation_observations;evaluation_predictions\n')

    def run(self):

        best_score = -np.inf
        early_stop_counter = 0

        for episode in range(config['training']['episodes']):

            observation = self.environment.reset()
            observation = observation[0]
            training_score, training_actions = run_training_episode(self.environment, self.agent, observation)

            evaluation_score, evaluation_actions, evaluation_observations, evaluation_predictions = run_evaluation_episodes(
                self.environment, self.agent)

            if np.average(evaluation_score) > best_score:
                if config['verbose']:
                    print(f'New best, saving model..')
                best_score = np.average(evaluation_score)
                self.agent.save_model(
                    f'{path}results/{self.created_at}/run_{self.run_num}/models/{best_score}.h5')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if config['verbose']:
                print(f'Episode: {episode}, Evaluation Score: {evaluation_score}, Training Score: {training_score}')
                print(self.agent.epsilon)

            save_results_to_file(self.created_at, self.run_num, episode, training_score, evaluation_score,
                                 self.agent.epsilon, training_actions, evaluation_actions, evaluation_observations,
                                 evaluation_predictions)

            if early_stop_counter >= config['training']['early_stop']:
                break

        self.environment.close()
