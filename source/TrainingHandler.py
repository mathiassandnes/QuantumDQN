import os
import numpy as np
import random
from datetime import datetime

import yaml

from source.Agent import Agent
from source.gym_handler import GymHandler
from source.utils import load_yml

config = load_yml('../configuration.yml')


def run_evaluation_episode(environment, agent, render=False):
    observation = environment.reset()
    observation = observation[0]
    total_reward = 0
    actions = []
    for step in range(config['training']['max_steps']):
        if config['environment']['render'] or render:
            environment.render()
        action = agent.choose_action(observation, training=False)
        actions.append(action)
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(actions)
    return total_reward


class TrainingHandler:
    def __init__(self, thread, trial):
        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = GymHandler()

        observation_space = self.environment.env.observation_space
        action_space = self.environment.env.action_space

        if config['mode'] == 'quantum':
            determine_quantum_parameters()

        self.agent = Agent(observation_space, action_space, config)

        self.init_result_files()

    def init_result_files(self):
        os.mkdir(f'../results/{self.created_at}')
        os.mkdir(f'../results/{self.created_at}/models')
        with open(f"../results/{self.created_at}/config.yml", "w+") as f:
            yaml.dump(config, f, default_flow_style=False)
        with open(f"../results/{self.created_at}/results.txt", "w+") as f:
            f.write('')

    def run(self):

        best_score = -np.inf

        for episode in range(config['training']['episodes']):
            observation = self.environment.reset()
            observation = observation[0]
            training_score = self.run_training_episode(observation)

            evaluation_score = run_evaluation_episode(self.environment, self.agent)

            if evaluation_score > best_score:
                print(f"New best, saving model..")
                best_score = evaluation_score
                self.agent.save_model(f"../results/{self.created_at}/models/{evaluation_score}.h5")

            print(f"Episode: {episode}, Evaluation Score: {evaluation_score}, Training Score: {training_score}")

            save_results_to_file(self.created_at, evaluation_score)
            print(self.agent.epsilon)

        self.environment.close()

    def run_training_episode(self, observation):

        # print(self.agent.model.get_weights())
        for timestep in range(config['training']['max_steps']):
            action = self.agent.choose_action(observation, training=True)

            next_observation, reward, terminated, truncated, info = self.environment.step(action)

            self.agent.remember(observation, action, reward, next_observation, terminated)
            self.agent.learn()

            observation = next_observation

            if config['environment']['render']:
                self.environment.render()

            if terminated or truncated:
                return timestep

        return config['training']['max_steps']


def save_results_to_file(created_at, score):
    with open(f"../results/{created_at}/results.txt", "r+") as f:
        history = f.read()
        history = history.split(',')
        history.append(str(score))
        f.seek(0)
        f.write(','.join(history))
        f.close()


def determine_quantum_parameters():
    bounds = config['quantum']['bounds']

    for key in config['quantum'].keys():
        if key == 'bounds':
            continue

        if key == 'rotations' or key == 'entanglements':
            config['quantum'][key] = [random.choice(bounds[key]) for _ in range(config['quantum']['layers'])]
            continue

        config['quantum'][key] = random.choice(bounds[key])
