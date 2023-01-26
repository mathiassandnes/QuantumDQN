import numpy as np
import pennylane as qml

from source.utils import load_yml

config = load_yml('../configuration.yml')


def get_circuit(n_layers, n_qubits, n_inputs):
    dev = qml.device('default.qubit.tf', wires=n_qubits)

    @qml.qnode(dev, interface='tf', diff_method='backprop')
    def circuit(inputs, weights):

        weights_per_layer = n_qubits * 2
        for layer_id in range(n_layers):

            # re-uploading
            for i in range(n_qubits):
                if i < n_inputs:
                    qml.RX(inputs[i], wires=i)
                else:
                    qml.Hadamard(wires=i)

            # Pauli X Parameterization
            for j in range(n_qubits):
                weight_id = layer_id * weights_per_layer + j
                qml.RX(weights[weight_id], wires=j)

            # Pauli Y Parameterization
            for k in range(n_qubits):
                weight_id = layer_id * weights_per_layer + k + n_qubits
                qml.RY(weights[weight_id], wires=k)

            # Entanglement
            for l in range(n_qubits - 1):
                qml.CNOT(wires=[l, l + 1])

        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    return circuit


def preprocess_observation(observation):
    position = observation[0]
    velocity = observation[1]

    position = (position + 1.2) / 1.8 * np.pi
    velocity = (velocity + 0.07) / 0.14 * np.pi

    position = round(position, 2)
    velocity = round(velocity, 2)

    observation = np.array([position, velocity])
    return observation
