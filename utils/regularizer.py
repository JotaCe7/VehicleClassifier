from tensorflow.keras import regularizers

def create_regularizer(layer_regularizers: dict):
  """
  Creates regularizer object to pass as an argument to
  keras.layers.Dense.
  Arguments:
  ----------
  layer_regularizer: dict
      Dictionary with all the options to create a regularizer keras
      object
  
  Returns:
  -------
      Dictionary ready to unpack and use as 'regularizer' argument in
      keras.layers.Dense object
  """
  regularizer = {}
  for key, value in layer_regularizers.items():
    # Instantiate class from module, get class name from dicrionary keys
    regularizer[key] = getattr(regularizers, list(value)[0])(**value[list(value)[0]])
  return regularizer

