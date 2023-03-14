import os
import gym

from source.Agent import Agent
from source.utils import load_yml


def find_best_model():
    model_names = os.listdir(f'{directory}/models')
    model_names = [name for name in model_names if name.endswith('.h5')]
    model_names = [''.join(name.split('.')[0:1]) for name in model_names]
    model_names = [float(name) for name in model_names]
    model_names = sorted(model_names)
    model_name = model_names[-1]

    return model_name


if __name__ == '__main__':
    directory = '../results/2023-03-13_11-56-31_thread_1_trial_0/run_0'
    config = load_yml(f'{directory}/../config.yml')

    environment = gym.make(config['environment']['name'], render_mode='human')

    model_path = find_best_model()
    model_path = f'{directory}/models/{model_path}.h5'
    agent = Agent(environment.observation_space, environment.action_space, config, model_path)

    episode_rewards = []
    for episode in range(10):
        observation = environment.reset()
        observation = observation[0]
        total_reward = 0
        actions = []
        for step in range(config['training']['max_steps']):
            environment.render()
            action = agent.choose_action(observation, training=False)
            actions.append(action)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f'Episode {episode} reward: {total_reward}')
        print(f'Actions: {actions}')
        episode_rewards.append(total_reward)

    environment.close()
    print(f'Average reward: {sum(episode_rewards) / len(episode_rewards)}')
