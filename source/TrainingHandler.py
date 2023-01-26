from datetime import datetime

from source.Agent import Agent
from source.gym_handler import GymHandler
from source.utils import load_yml

config = load_yml('../configuration.yml')


def run_evaluation_episode(environment, agent):
    observation = environment.reset()
    total_reward = 0
    for step in range(config['training']['max_steps']):
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated = environment.step(action)
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

    def run(self):

        history = []
        for episode in range(config['training']['episodes']):
            observation = self.environment.reset()
            observation = observation[0]

            self.run_training_episode(observation)

            score = run_evaluation_episode(self.environment, self.agent)

            print(f"Episode: {episode}, Score: {score}")
            history.append(score)
        self.environment.close()

        # write history to file
        with open(f"../results/{self.created_at}.txt", "w+") as f:
            f.write(str(history))
            f.close()

    def run_training_episode(self, observation):
        for timestep in range(config['training']['max_steps']):
            if config['verbose']:
                print(f'Timestep: {timestep}')
            action = self.agent.choose_action(observation)

            next_observation, reward, terminated, truncated, info = self.environment.step(action)

            self.agent.remember(observation, action, reward, next_observation, terminated)
            self.agent.learn()

            observation = next_observation

            if config['environment']['render']:
                self.environment.render()

            if terminated or truncated:
                break
