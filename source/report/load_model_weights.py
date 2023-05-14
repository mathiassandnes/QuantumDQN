import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('500.h5')

# Loop through all layers
for layer in model.layers:
    # Get the weights of the layer
    weights = layer.get_weights()

    # If the layer has weights
    if weights:
        print(f"Layer {layer.name} weights :")
        print(weights)
        print("\n")
    else:
        print(f"Layer {layer.name} has no weights.\n")