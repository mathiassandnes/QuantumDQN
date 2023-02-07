import numpy as np
import pennylane as qml

from source.utils import load_yml

config = load_yml('../configuration.yml')


def get_circuit(n_layers, n_qubits, n_inputs):
    dev = qml.device('default.qubit.tf', wires=n_qubits)

    @qml.qnode(dev, interface='tf', diff_method='backprop')
    def circuit(inputs, weights):

        weight_index = 0
        for layer_index in range(n_layers):

            # Embedding
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                if i < n_inputs:
                    qml.RX(inputs[i], wires=i)
                # else:
                #     qml.Hadamard(wires=i)

            # Rotations
            weight_index = rotate(layer_index, weights, weight_index)

            # Entanglement
            entangle(layer_index)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def rotate(layer_index, weights, weight_index):

        for rotation in config['quantum']['rotations'][layer_index]:
            match rotation:
                case 'x':
                    for i in range(n_qubits):
                        qml.RX(weights[weight_index], wires=i)
                        weight_index += 1
                case 'y':
                    for i in range(n_qubits):
                        qml.RY(weights[weight_index], wires=i)
                        weight_index += 1
                case 'z':
                    for i in range(n_qubits):
                        qml.RZ(weights[weight_index], wires=i)
                        weight_index += 1

        return weight_index

    def entangle(layer_index):
        match config['quantum']['entanglements'][layer_index]:
            case 'ladder':
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            case 'double ladder':
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits - 1, 0, -1):
                    qml.CNOT(wires=[i, i - 1])
            case 'full':
                for i in range(n_qubits - 1):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            case 'none':
                pass
            case 'brick':
                brick_layer_type = 0
                for _ in range(config['quantum']['brick_size']):
                    if brick_layer_type == 0:
                        for i in range(0, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                    if brick_layer_type == 1:
                        for i in range(1, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                    brick_layer_type = 1 - brick_layer_type

    return circuit


def preprocess_mountaincar(observation):
    position = observation[0]
    velocity = observation[1]

    position = (position + 1.2) / 1.8 * np.pi
    velocity = (velocity + 0.07) / 0.14 * np.pi

    position = round(position, 2)
    velocity = round(velocity, 2)

    observation = np.array([position, velocity])
    return observation


def preprocess_cartpole(observation):
    position = observation[0]
    velocity = observation[1]
    angle = observation[2]
    angular_velocity = observation[3]

    position = (position + 2.4) / 4.8 * np.pi
    velocity = (velocity + 0.5) / 1 * np.pi
    angle = (angle + 0.2094) / 0.4188 * np.pi
    angular_velocity = (angular_velocity + 1) / 2 * np.pi

    position = round(position, 2)
    velocity = round(velocity, 2)
    angle = round(angle, 2)
    angular_velocity = round(angular_velocity, 2)

    observation = np.array([position, velocity, angle, angular_velocity])
    return observation


def preprocess_observation(observation):
    match config['environment']['name']:
        case 'CartPole-v1':
            return preprocess_cartpole(observation)
        case 'MountainCar-v0':
            return preprocess_mountaincar(observation)
