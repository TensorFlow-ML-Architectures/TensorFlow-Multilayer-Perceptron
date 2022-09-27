from keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np

class Mlp:
    def __int__(self):
        self.dims = None
        self.num_classes = None
        self.model = None
        self.inputs = None

    def model_create(self, dims, num_classes):
        self.dims = dims
        self.num_classes = num_classes
        self.inputs = keras.Input(shape=self.dims)

        # Image augmentation block
        rot = 0.1
        flip = "horizontal"
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip(flip),
                layers.RandomRotation(rot),
            ]
        )
        x = data_augmentation(self.inputs)

        # re-scale
        x = layers.Rescaling(1.0 / 255)(x)

        # re-shape
        x = layers.Reshape([-1, np.prod(self.dims)])(x)
        x = keras.backend.squeeze(x=x, axis=1)

        for size in [128, 256, 512, 728, 1024]:
            x = layers.Dense(size, activation="relu")(x)
            x = layers.BatchNormalization()(x)

        units = None
        if self.num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = self.num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)

        self.model = keras.Model(self.inputs, outputs)

        self.model.summary()