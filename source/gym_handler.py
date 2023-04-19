import gym

from source.utils import load_yml

config = load_yml('../configuration.yml')

path = '' if config['cluster'] else '../'


class GymHandler:
    def __init__(self):
        self.seed = config['random_seed']
        if config['environment']['render']:
            self.env = gym.make(config['environment']['name'], render_mode=config['environment']['render_mode'])
        else:
            self.env = gym.make(config['environment']['name'])

            self.env.reset()
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def update_seed(self):
        self.seed += 1

    def reset(self):
        self.update_seed()
        return self.env.reset(seed=self.seed)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
