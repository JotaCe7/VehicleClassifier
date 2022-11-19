from tensorflow.keras import regularizers

def create_regularizer(layer_regularizers: dict):
  regularizer = {}
  for key, value in layer_regularizers.items():
    # Instantiate class from module, get class name from dicrionary keys
    regularizer[key] = getattr(regularizers, list(value)[0])(**value[list(value)[0]])
  return regularizer

