from utils.data_aug import create_data_aug_layer


from tensorflow import keras
import tensorflow as tf
from keras.layers import BatchNormalization

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

def set_n_last_layers(layers, nlayers: int, trainable : bool = True):
  nLayers = 0
  for layer in layers:
    nLayers +=1
    if (nLayers > (len(layers) -nlayers)):
      layer.trainable = False if isinstance(layer, BatchNormalization) else True
    else:
      layer.trainable = False 



def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
):
    
    # Create the model to be used for finetuning here!
    if weights == "imagenet":
        # Define the Input layer
        # Assign it to `input` variable
        # Use keras.layers.Input(), following this requirements:
        #   1. layer dtype must be tensorflow.float32

        input = keras.layers.Input(shape=input_shape)
        input = tf.cast(input, tf.float32)

        # Create the data augmentation layers here and add to the model next
        # to the input layer
        # If no data augmentation was used, skip this
        #bach = keras.layers.BatchNormalization()(input)
        if data_aug_layer != None:

            data_aug = create_data_aug_layer(data_aug_layer)
            input = data_aug(input)

        #bach2 = keras.layers.BatchNormalization()(norm)

        # Add a layer for preprocessing the input images values
        # E.g. change pixels interval from [0, 255] to [0, 1]
        # Resnet50 already has a preprocessing function you must use here
        # See keras.applications.resnet50.preprocess_input()

        prepro = keras.applications.resnet50.preprocess_input(input)
        # Create the corresponding core model using
        # keras.applications.ResNet50()
        # The model created here must follow this requirements:
        #   1. Use imagenet weights
        #   2. Drop top layer (imagenet classification layer)
        #   3. Use Global average pooling as model output

        core_model = keras.applications.resnet50.ResNet50(
                                      weights="imagenet",
                                      include_top=False,
                                      pooling="avg"
                                      )
        core_model.trainable = False
        model = core_model(prepro, training=False)

        model = keras.layers.Dropout(dropout_rate)(model)
        # Add the classification layer here, use keras.layers.Dense() and
        # `classes` parameter
        # Assign it to `outputs` variable
        from tensorflow.keras import regularizers

        kernel_regularizer = regularizers.L1L2(l1=1e-5, l2=1e-4)

        outputs = keras.layers.Dense(classes, kernel_regularizer=regularizers.L1(1e-4), activation='softmax')(model)
        # kernel_regularizer=regularizers.L1L2(
        # l1=1e-5, l2=1e-4),
        # Now you have all the layers in place, create a new model
        # Use keras.Model()
        # Assign it to `model` variable

        model = keras.Model(input, outputs)
    else:
        # For this particular case we want to load our already defined and
        # finetuned model, see how to do this using keras
        # Assign it to `model` variable
        # TODO
        model = tf.keras.models.load_model(weights)

    return model