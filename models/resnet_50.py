from tensorflow import keras, float32
from tensorflow.keras.applications import resnet50
from utils.data_aug import create_data_aug_layer
from utils.regularizer import create_regularizer


def get_number_of_trainable(layers):
    trainables = 0
    not_trainables = 0
    for layer in layers:
      if layer.trainable:
        trainables = trainables + 1
      else:
        not_trainables = not_trainables + 1
    
    print('*****************************')
    print('Trainable layers:', trainables)
    print('Not trainable layers:', not_trainables)
    return trainables, not_trainables

def unfreeze_n_last_layers(layers, nlayers: int, trainable : bool = True):
  nLayers = 0
  for layer in layers:
    nLayers +=1
    if (nLayers > (len(layers) -nlayers)):
      layer.trainable = False if isinstance(layer, keras.layers.BatchNormalization) else True
      #layer.trainable = True
    else:
      layer.trainable = False 

def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = 196,
    regularizers: dict = {},
    output_regularizer: dict = {},
    trainable: bool = False,
    n_dense_layers=0,
    n_unfreeze_layers=0
):
    """
    Creates and loads the Resnet50 model we will use for our experiments.
    Depending on the `weights` parameter, this function will return one of
    two possible keras models:
        1. weights='imagenet': Returns a model ready for performing finetuning
                               on your custom dataset using imagenet weights
                               as starting point.
        2. weights!='imagenet': Then `weights` must be a valid path to a
                                pre-trained model on our custom dataset.
                                This function will return a model that can
                                be used to get predictions on our custom task.

    See an extensive tutorial about finetuning with Keras here:
    https://www.tensorflow.org/tutorials/images/transfer_learning.

    Parameters
    ----------
    weights : str
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size cannot be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    dropout_rate : float
        Value used for Dropout layer to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    data_aug_layer : dict
        Configuration from experiment YAML file used to setup the data
        augmentation process during finetuning.
        Only needed when weights='imagenet'.

    classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """

    if weights == "imagenet":
        # Define the Input layer
        input = keras.layers.Input(shape=(input_shape), dtype=float32)

        # Add augmentation layer
        #x = create_data_aug_layer(data_aug_layer)(input) if data_aug_layer else input
        if data_aug_layer is not None:
          input = create_data_aug_layer(data_aug_layer)(input)

        # Add a layer for preprocessing the input images values
        x = resnet50.preprocess_input(input)

        # Instantiate ResNet50 architecture
        core_model = resnet50.ResNet50(
                                        weights='imagenet',       # Load weights pre*trained on ImageNet
                                        #input_shape=input_shape,  # image shape
                                        include_top=False,        # Do not include tehe ImageNet classifier at the top
                                        pooling="avg"             # gloval average pooling
                                      )
        if n_unfreeze_layers > 0:
          unfreeze_n_last_layers(core_model.layers, n_unfreeze_layers, trainable)
        else:
          core_model.trainable = trainable # Freeze core model or not

        x = core_model(x, training = False)

        # Add a single dropout layer for regularization
        x = keras.layers.Dropout(dropout_rate)(x)

        if bool(regularizers):
          regularizers = create_regularizer(regularizers)



        while n_dense_layers>1:
          # Add a dense layer
          x = keras.layers.Dense(2**(7+n_dense_layers),
                                activation='relu',
                                **regularizers)(x)
          # Add a single dropout layer for regularization
          x = keras.layers.Dropout(dropout_rate)(x)
          n_dense_layers-=1

        if bool(output_regularizer):
          output_regularizer = create_regularizer(output_regularizer)



        # Add classification layer
        outputs = keras.layers.Dense(classes,
                                     activation='softmax',
                                     **output_regularizer)(x)

        # Create the model
        model = keras.Model(input, outputs)
    else:
        # Load our already defined and finetuned model,
        model = keras.models.load_model(weights)

    return model
