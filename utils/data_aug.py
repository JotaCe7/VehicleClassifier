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

    # for key, value in data_aug_layer.items():
    #   # Instantiate clsas from module, get class name from dicrionary keys
    #   Layer = getattr(layers, key.replace("_"," ").title().replace(" ",""))
    #   data_aug_layers.append(Layer(**value))
    
    if "random_flip" in data_aug_layer:
      data_aug_layers.append(layers.RandomFlip(**data_aug_layer["random_flip"]))

    if 'random_rotation' in data_aug_layer:
        data_aug_layers.append(layers.RandomRotation(**data_aug_layer['random_rotation']))

    if 'random_zoom' in data_aug_layer:
        data_aug_layers.append(layers.RandomZoom(**data_aug_layer['random_zoom']))

    # Return a keras.Sequential model having the new layers created
    data_augmentation = Sequential(layers=data_aug_layers)

    return data_augmentation
