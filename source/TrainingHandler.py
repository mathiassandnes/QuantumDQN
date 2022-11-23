from datetime import datetime

from source.Agent import Agent
from source.gym_handler import GymHandler
from source.utils import load_yml

config = load_yml('../configuration.yml')


class TrainingHandler:
    def __init__(self):
        self.created_at = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.environment = GymHandler()
        self.agent = Agent(self.environment.env.action_space)

    def run(self):

        for episode in range(config['training']['episodes']):
            score = 0
            observation, info = self.environment.reset()

            for _ in range(config['training']['max_steps']):

                action = self.agent.choose_action(observation)

                next_observation, reward, terminated, truncated, info = self.environment.step(action)

                self.agent.remember(observation, action, reward, next_observation, terminated or truncated)
                self.agent.learn()
                observation = next_observation
                score += reward

                if terminated or truncated:
                    break

            print(f"Episode: {episode}, Score: {score}, Info: {info}")
        self.environment.close()
