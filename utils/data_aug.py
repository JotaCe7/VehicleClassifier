from tensorflow.keras import layers, Sequential

def create_data_aug_layer(data_aug_layer: dict):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    
    data_aug_layers = []

    for key, value in data_aug_layer.items():
      # Instantiate clsas from module, get class name from dicrionary keys
      Layer = getattr(layers, key.replace("_"," ").title().replace(" ",""))
      layer = Layer()
      # Append the data augmentation layers on this list
      data_aug_layers.append(layer(value))

    # Return a keras.Sequential model having the new layers created
    data_augmentation = Sequential(layers=data_aug_layers)

    return data_augmentation
