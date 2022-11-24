import pennylane as qml
import numpy as np
import tensorflow as tf

from source.utils import load_yml
from source.ReplayBuffer import ReplayBuffer

config = load_yml('../configuration.yml')


class Agent:
    def __init__(self, action_space):
        self.memory = ReplayBuffer()
        self.action_space = action_space
        self.epsilon = config['training']['epsilon']['start']

        self.model = self.build_model()

    def get_random_action(self):
        action = self.action_space.sample()
        return action

    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            return self.get_random_action()

        observation = preprocess_observation(observation)
        prediction = self.model.predict(observation, verbose=0)
        action = np.argmax(prediction)
        return action

    def remember(self, observation, action, reward, next_observation, terminated):
        self.memory.store(observation, action, reward, next_observation, terminated)

    def learn(self):
        if self.memory.memory_counter < config['training']['batch_size']:
            return
        batch_size = config['training']['batch_size']
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.memory.sample_batch(
            batch_size)

        action_space = np.arange(config['number_of_actions'])
        action_values = np.array(action_space, dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        action_indices = [int(x) for x in action_indices]

        q_eval = self.model.predict(state_batch, verbose=0)
        q_next = self.model.predict(next_state_batch, verbose=0)

        batch_index = np.arange(batch_size, dtype=np.int32)

        gradients = reward_batch + np.max(q_next, axis=1) * (1 - terminated_batch)
        q_eval[batch_index, action_indices] = gradients / 30

        self.model.fit(state_batch, q_eval, verbose=0)

        # if self.exploration_rate > self.exploration_rate_min:
        #     self.exploration_rate *= self.exploration_rate_decrement

    def build_model(self):
        q_circuit = get_circuit()
        weight_shapes = {'weights': 12}
        qlayer = qml.qnn.KerasLayer(q_circuit, weight_shapes, output_dim=3)
        clayer = tf.keras.layers.Dense(3)

        self.model = tf.keras.models.Sequential([qlayer, clayer])

        self.model.compile(loss=config['training']['loss'],
                           optimizer=config['training']['optimizer'])
        self.model.build((None, 2))

        return self.model


def get_circuit():
    dev = qml.device('default.qubit.tf', wires=config['qubits'])

    @qml.qnode(dev, interface='tf', diff_method='backprop')
    def circuit(inputs, weights):
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)
        qml.Hadamard(wires=2)

        qml.RX(weights[0], wires=0)
        qml.RX(weights[1], wires=1)
        qml.RX(weights[2], wires=2)

        qml.RY(weights[3], wires=0)
        qml.RY(weights[4], wires=1)
        qml.RY(weights[5], wires=2)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)
        qml.Hadamard(wires=2)

        qml.RX(weights[6], wires=0)
        qml.RX(weights[7], wires=1)
        qml.RX(weights[8], wires=2)

        qml.RY(weights[9], wires=0)
        qml.RY(weights[10], wires=1)
        qml.RY(weights[11], wires=2)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliX(2))

    return circuit


def preprocess_observation(observation):
    position = observation[0]
    velocity = observation[1]

    position = (position + 1.2) / 1.8 * np.pi
    velocity = (velocity + 0.07) / 0.14 * np.pi

    position = round(position, 2)
    velocity = round(velocity, 2)

    observation = np.array([position, velocity]).reshape(1, 2)
    return observation


if __name__ == '__main__':
    agent = Agent(None)
    agent.choose_action([0.0, 0.0])
