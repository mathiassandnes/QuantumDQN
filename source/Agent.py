import pennylane as qml
import numpy as np
import tensorflow as tf

from source.quantum_utils import get_circuit, preprocess_observation
from source.utils import load_yml
from source.ReplayBuffer import ReplayBuffer

config = load_yml('../configuration.yml')


class Agent:
    def __init__(self, observation_space, action_space):
        self.n_inputs = observation_space.shape[0]
        self.n_outputs = action_space.n
        self.memory = ReplayBuffer(self.n_inputs, self.n_outputs)
        self.action_space = action_space
        self.epsilon = config['training']['epsilon']['start']

        mode = config['mode']
        match mode:
            case 'quantum':
                self.model = self.build_quantum_model()
            case 'classical':
                self.model = self.build_classical_model()

    def get_random_action(self):
        action = self.action_space.sample()
        return action

    def choose_action(self, observation, training=True):
        if np.random.rand() < self.epsilon and training:
            return self.get_random_action()

        if config['mode'] == 'quantum':
            observation = preprocess_observation(observation)

        observation = observation.reshape(1, self.n_inputs)

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

        action_space = np.arange(self.n_outputs)
        action_values = np.array(action_space, dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        action_indices = [int(x) for x in action_indices]

        q_eval = self.model.predict(state_batch, verbose=0)
        q_next = self.model.predict(next_state_batch, verbose=0)

        batch_index = np.arange(batch_size, dtype=np.int32)

        gradients = reward_batch + tf.reduce_max(q_next, axis=1) * (1 - terminated_batch)
        q_eval[batch_index, action_indices] = gradients

        self.model.fit(state_batch, q_eval, verbose=0)

        # if self.exploration_rate > self.exploration_rate_min:
        #     self.exploration_rate *= self.exploration_rate_decrement

    def build_quantum_model(self):
        n_layers = config['quantum']['layers']
        n_qubits = config['quantum']['qubits']
        n_weights = 2 * n_layers * n_qubits

        if n_qubits < self.n_inputs:
            raise 'The number of inputs is larger than number of Qubits'

        if n_qubits < self.n_outputs:
            raise 'The number of outputs is larger than number of Qubits'

        model = tf.keras.models.Sequential()

        q_circuit = get_circuit(n_layers, n_qubits, self.n_inputs)

        weights = {'weights': n_weights}
        model.add(qml.qnn.KerasLayer(q_circuit, weights, output_dim=n_qubits))

        model.add(tf.keras.layers.Dense(self.n_outputs))

        model.compile(loss=config['training']['loss'],
                      optimizer=config['training']['optimizer'])
        model.build((None, self.n_inputs))
        if config['verbose']:
            model.summary()
            drawer = qml.draw(q_circuit)
            print(drawer(inputs=[0.0, 0.0], weights=np.arange(n_weights)))
        return model

    def build_classical_model(self):
        n_layers = config['classical']['layers']
        n_neurons = config['classical']['neurons']

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Input(self.n_inputs))

        for _ in range(n_layers):
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))

        model.add(tf.keras.layers.Dense(self.n_outputs, activation='linear'))

        model.compile(loss=config['training']['loss'],
                      optimizer=config['training']['optimizer'])
        model.build((None, self.n_outputs))
        if config['verbose']:
            model.summary()

        return model


if __name__ == '__main__':
    agent = Agent([None, None], [None, None, None, None])
    pred = agent.choose_action([0.0, 0.0])
