import numpy as np
import pennylane as qml

from source.utils import load_yml

config = load_yml('../configuration.yml')
path = '' if config['cluster'] else '../'


def get_circuit(n_layers, n_qubits, n_inputs, hyperparameters):
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

            if config['quantum']['trainable_entanglements']:
                weight_index = trainable_entangle(layer_index, weights, weight_index)
            else:
                entangle(layer_index)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    def rotate(layer_index, weights, weight_index):
        for rotation in hyperparameters['rotations'][layer_index]:
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
        match hyperparameters['entanglements'][layer_index]:
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
                for _ in range(hyperparameters['brick_size']):
                    if brick_layer_type == 0:
                        for i in range(0, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                    if brick_layer_type == 1:
                        for i in range(1, n_qubits - 1, 2):
                            qml.CNOT(wires=[i, i + 1])
                    brick_layer_type = 1 - brick_layer_type

    def trainable_entangle(layer_index, weights, weight_index):
        entanglement_strengths = []
        match hyperparameters['entanglements'][layer_index]:
            case 'ladder':
                for i in range(n_qubits - 1):
                    qml.CRX(weights[weight_index], wires=[i, i + 1])
                    entanglement_strengths.append(weights[weight_index])
                    weight_index += 1
            case 'full':
                k = 0
                for i in range(n_qubits - 1):
                    for j in range(i + 1, n_qubits):
                        qml.CRX(weights[weight_index], wires=[i, j])
                        entanglement_strengths.append(weights[weight_index])
                        weight_index += 1
                        k += 1
        return weight_index

    return circuit


def preprocess_mountaincar(observation):
    position = observation[0]
    velocity = observation[1]

    position = (position + 1.2) / 1.8 * np.pi - np.pi / 2
    velocity = (velocity + 0.07) / 0.14 * np.pi - np.pi / 2

    position = round(position, 2)
    velocity = round(velocity, 2)

    observation = np.array([position, velocity])
    return observation


def preprocess_acrobot(observation):
    costheta1 = observation[0]
    sintheta1 = observation[1]
    costheta2 = observation[2]
    sintheta2 = observation[3]
    angular_velocity_theta1 = observation[4]
    angular_velocity_theta2 = observation[5]

    costheta1 = (costheta1 + 1) / 2 * np.pi - np.pi / 2
    sintheta1 = (sintheta1 + 1) / 2 * np.pi - np.pi / 2
    costheta2 = (costheta2 + 1) / 2 * np.pi - np.pi / 2
    sintheta2 = (sintheta2 + 1) / 2 * np.pi - np.pi / 2
    angular_velocity_theta1 = (angular_velocity_theta1 + 4) / 8 * np.pi - np.pi / 2
    angular_velocity_theta2 = (angular_velocity_theta2 + 9) / 18 * np.pi - np.pi / 2

    costheta1 = round(costheta1, 2)
    sintheta1 = round(sintheta1, 2)
    costheta2 = round(costheta2, 2)
    sintheta2 = round(sintheta2, 2)
    angular_velocity_theta1 = round(angular_velocity_theta1, 2)
    angular_velocity_theta2 = round(angular_velocity_theta2, 2)

    observation = np.array(
        [costheta1, sintheta1, costheta2, sintheta2, angular_velocity_theta1, angular_velocity_theta2])
    return observation


def preprocess_cartpole(observation):
    position = observation[0]
    velocity = observation[1]
    angle = observation[2]
    angular_velocity = observation[3]

    position = (position + 2.4) / 4.8 * np.pi - np.pi / 2
    velocity = (velocity + 0.5) / 1 * np.pi - np.pi / 2
    angle = (angle + 0.2094) / 0.4188 * np.pi - np.pi / 2
    angular_velocity = (angular_velocity + 1) / 2 * np.pi - np.pi / 2

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

    position_x = (position_x + 1) / 2 * np.pi - np.pi / 2
    position_y = (position_y + 1) / 2 * np.pi - np.pi / 2
    velocity_x = (velocity_x + 1) / 2 * np.pi - np.pi / 2
    velocity_y = (velocity_y + 1) / 2 * np.pi - np.pi / 2
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


def count_entanglement_gates(qubits, scheme):
    count = 0
    for s in scheme:
        match s:
            case 'ladder':
                count += (qubits - 1)
            case 'double ladder':
                count += 2 * (qubits - 1)
            case 'full':
                count += (qubits * (qubits - 1)) // 2
            case 'none':
                pass

    return count


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


def get_n_weights(entanglements, n_qubits, rotations, trainable_entanglements):
    rotations_per_qubit = len([item for sublist in rotations for item in sublist])
    n_weights = rotations_per_qubit * n_qubits
    if trainable_entanglements:
        n_weights += count_entanglement_gates(n_qubits, entanglements)
    return n_weights
