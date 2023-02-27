import numpy as np
import pennylane as qml

from source.utils import load_yml

config = load_yml('../configuration.yml')


class MyKerasLayer(qml.qnn.KerasLayer):
    def get_config(self):
        config_ = super().get_config()
        config_['qnode'] = self.qnode
        return config_


def get_circuit(n_layers, n_qubits, n_inputs, _config):
    global config
    config = _config
    dev = qml.device('default.qubit.tf', wires=n_qubits)

    @qml.qnode(dev, interface='tf', diff_method='best')
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


def preprocess_acrobot(observation):
    costheta1 = observation[0]
    sintheta1 = observation[1]
    costheta2 = observation[2]
    sintheta2 = observation[3]
    theta1dot = observation[4]
    theta2dot = observation[5]

    costheta1 = (costheta1 + 1) / 2 * np.pi
    sintheta1 = (sintheta1 + 1) / 2 * np.pi
    costheta2 = (costheta2 + 1) / 2 * np.pi
    sintheta2 = (sintheta2 + 1) / 2 * np.pi
    theta1dot = (theta1dot + 4) / 8 * np.pi
    theta2dot = (theta2dot + 9) / 18 * np.pi

    costheta1 = round(costheta1, 2)
    sintheta1 = round(sintheta1, 2)
    costheta2 = round(costheta2, 2)
    sintheta2 = round(sintheta2, 2)
    theta1dot = round(theta1dot, 2)
    theta2dot = round(theta2dot, 2)

    observation = np.array([costheta1, sintheta1, costheta2, sintheta2, theta1dot, theta2dot])
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


def preprocess_lunar_lander(observation):
    position_x = observation[0]
    position_y = observation[1]
    velocity_x = observation[2]
    velocity_y = observation[3]
    angle = observation[4]
    angular_velocity = observation[5]
    leg1_contact = observation[6]
    leg2_contact = observation[7]

    position_x = (position_x + 1) / 2 * np.pi
    position_y = (position_y + 1) / 2 * np.pi
    velocity_x = (velocity_x + 1) / 2 * np.pi
    velocity_y = (velocity_y + 1) / 2 * np.pi
    angle = (angle + 1) / 2 * np.pi
    angular_velocity = (angular_velocity + 1) / 2 * np.pi
    leg1_contact = (leg1_contact + 1) / 2 * np.pi
    leg2_contact = (leg2_contact + 1) / 2 * np.pi

    position_x = round(position_x, 2)
    position_y = round(position_y, 2)
    velocity_x = round(velocity_x, 2)
    velocity_y = round(velocity_y, 2)
    angle = round(angle, 2)
    angular_velocity = round(angular_velocity, 2)
    leg1_contact = round(leg1_contact, 2)
    leg2_contact = round(leg2_contact, 2)

    observation = np.array(
        [position_x, position_y, velocity_x, velocity_y, angle, angular_velocity, leg1_contact, leg2_contact])
    return observation


def preprocess_observation(observation):
    match config['environment']['name']:
        case 'CartPole-v1':
            return preprocess_cartpole(observation)
        case 'MountainCar-v0':
            return preprocess_mountaincar(observation)
        case 'Acrobot-v1':
            return preprocess_acrobot(observation)
        case 'LunarLander-v2':
            return preprocess_lunar_lander(observation)
