import pennylane as qml
import numpy as np
import tensorflow as tf

from source.quantum_utils import get_circuit, preprocess_observation, count_entanglement_gates
from source.utils import load_yml
from source.ReplayBuffer import ReplayBuffer

config = load_yml(f'../configuration.yml')
path = '' if config['cluster'] else '../'


class Agent:
    def __init__(self, observation_space, action_space, hyperparameters=None, model=None):
        self.hyperparameters = hyperparameters
        self.n_inputs = observation_space.shape[0]
        self.n_outputs = action_space.n
        self.memory = ReplayBuffer(self.n_inputs, self.n_outputs)
        self.action_space = action_space
        self.epsilon = config['training']['epsilon']['start']
        self.exploration_rate_min = config['training']['epsilon']['end']
        self.exploration_rate_decrement = config['training']['epsilon']['change']

        mode = config['mode']
        match mode:
            case 'quantum':
                self.model = self.build_quantum_model(model)
            case 'classical':
                self.model = self.build_classical_model(model)

    def save_model(self, path):
        self.model.save(path)

    def get_random_action(self):
        action = self.action_space.sample()
        return action

    def choose_action(self, observation, training=True, include_raw=False):

        if np.random.rand() < self.epsilon and training:
            return self.get_random_action()

        if config['mode'] == 'quantum':
            observation = preprocess_observation(observation)

        observation = observation.reshape(1, self.n_inputs)

        prediction = self.model.predict(observation, verbose=0)

        action = np.argmax(prediction)

        if include_raw:
            return action, prediction

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

        if self.epsilon > self.exploration_rate_min:
            self.epsilon *= self.exploration_rate_decrement

    def build_quantum_model(self, model_path=None):

        n_layers = self.hyperparameters['layers']
        n_qubits = self.hyperparameters['qubits']
        rotations = self.hyperparameters['rotations']
        entanglements = self.hyperparameters['entanglements']

        rotations_per_qubit = len([item for sublist in rotations for item in sublist])
        n_weights = rotations_per_qubit * n_qubits
        if config['quantum']['trainable_entanglements']:

            n_weights += count_entanglement_gates(n_qubits, entanglements)

        if n_qubits < self.n_inputs:
            print(f'Inputs: {self.n_inputs}, Qubits: {n_qubits}')
            raise 'The number of inputs is larger than number of Qubits.'

        print(
            f'Qubits: {n_qubits}, '
            f'Weights: {n_weights}, '
            f'Layers: {n_layers}, '
            f'Rotations: {rotations}, '
            f'Entanglements: {entanglements}')
        weight_shapes = {'weights': n_weights}
        q_circuit = get_circuit(n_layers, n_qubits, self.n_inputs, self.hyperparameters)

        q_layer = qml.qnn.KerasLayer(q_circuit, weight_shapes, output_dim=n_qubits)
        output = tf.keras.layers.Dense(self.n_outputs, activation='linear')

        model = tf.keras.models.Sequential([q_layer, output])

        opt = tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate'])

        model.compile(opt, loss='mae')
        if config['verbose']:
            excluded_keys = ['bounds']
            keys_except_excluded = {key: value for key, value in self.hyperparameters.items() if
                                    key not in excluded_keys}
            print(keys_except_excluded)
            drawer = qml.draw(q_circuit)
            print(drawer(inputs=np.arange(self.n_inputs), weights=np.arange(n_weights)))
        return model

    def build_classical_model(self, model=None):
        if model:
            return tf.keras.models.load_model(model)

        n_layers = self.hyperparameters['layers']
        n_neurons = self.hyperparameters['neurons']

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Input(self.n_inputs))

        for _ in range(n_layers):
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))

        model.add(tf.keras.layers.Dense(self.n_outputs, activation='linear'))

        model.compile(loss=config['training']['loss'],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']))
        model.build((None, self.n_outputs))
        if config['verbose']:
            model.summary()

        return model


if __name__ == '__main__':
    agent = Agent([None, None], [None, None, None, None])
    for _ in range(100):
        input = np.random.rand(4)
        pred = agent.choose_action(input, training=False)
        print(pred)
