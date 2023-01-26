import numpy as np

from utils import reverse_one_hot, load_yml

config = load_yml('../configuration.yml')


class ReplayBuffer:
    def __init__(self, n_inputs, n_outputs):
        self.memory_size = config['memory_size']
        self.input_dimensions = n_inputs
        self.number_of_actions = n_outputs
        self.memory_counter = 0

        self.state_memory = np.zeros((self.memory_size, self.input_dimensions))
        self.next_state_memory = np.zeros((self.memory_size, self.input_dimensions))
        self.action_memory = np.zeros((self.memory_size, self.number_of_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminated_memory = np.zeros(self.memory_size)

    def store(self, state, action, reward, next_state, terminated):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminated_memory[index] = terminated

        action = reverse_one_hot(action, self.number_of_actions)
        self.action_memory[index] = action
        self.memory_counter += 1

    def sample_batch(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch_indices = np.random.choice(max_memory, batch_size)

        state_batch = self.state_memory[batch_indices]
        action_batch = self.action_memory[batch_indices]
        reward_batch = self.reward_memory[batch_indices]
        next_state_batch = self.next_state_memory[batch_indices]
        terminated_batch = self.terminated_memory[batch_indices]

        return state_batch, action_batch, reward_batch, next_state_batch, terminated_batch
