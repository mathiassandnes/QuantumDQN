import gym

from source.utils import load_yml

config = load_yml('../configuration.yml')

path = '' if config['cluster'] else '../'

class GymHandler:
    def __init__(self):
        if config['environment']['render']:
            self.env = gym.make(config['environment']['name'], render_mode=config['environment']['render_mode'])
        else:
            self.env = gym.make(config['environment']['name'])

            self.env.reset()
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset(seed=config['random_seed'])

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


