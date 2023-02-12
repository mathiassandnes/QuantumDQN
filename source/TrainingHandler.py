import os
import numpy as np
from datetime import datetime

from source.Agent import Agent
from source.gym_handler import GymHandler
from source.utils import load_yml

config = load_yml('../configuration.yml')


def run_evaluation_episode(environment, agent):
    observation = environment.reset()
    observation = observation[0]
    total_reward = 0
    for step in range(config['training']['max_steps']):
        if config['environment']['render']:
            environment.render()
        action = agent.choose_action(observation, training=False)
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward


class TrainingHandler:
    def __init__(self):
        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = GymHandler()

        observation_space = self.environment.env.observation_space
        action_space = self.environment.env.action_space

        self.agent = Agent(observation_space, action_space)

        self.init_result_files()

    def init_result_files(self):
        os.mkdir(f'../results/{self.created_at}')
        with open(f"../results/{self.created_at}/config.yml", "w+") as f:
            f.write(str(config))
            f.close()
        with open(f"../results/{self.created_at}/results.txt", "w+") as f:
            f.write('')
            f.close()

    def run(self):

        if config['mode'] == 'quantum':
            determine_quantum_parameters()

        for episode in range(config['training']['episodes']):
            observation = self.environment.reset()
            observation = observation[0]

            self.run_training_episode(observation)

            score = run_evaluation_episode(self.environment, self.agent)

            print(f"Episode: {episode}, Score: {score}")

            save_results_to_file(self.created_at, score)

        self.environment.close()

    def run_training_episode(self, observation):
        # print model raw parameters
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
                break


def save_results_to_file(created_at, score):
    with open(f"../results/{created_at}/results.txt", "r+") as f:
        history = f.read()
        history = history.split(',')
        history.append(str(score))
        f.seek(0)
        f.write(','.join(history))
        f.close()


def determine_quantum_parameters():
    pass
