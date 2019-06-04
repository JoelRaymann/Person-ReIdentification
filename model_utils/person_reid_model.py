# For python 2 support
from __future__ import absolute_import, print_function, division, unicode_literals

# import necessary packages
import tensorflow as tf
from tensorflow import keras

# Person ReID Model
def person_recognition_model():
    """
    Functional API to define the recogniton model
    """
    x1 = keras.layers.Input(shape = (299, 299, 3), name = "input_1", dtype = "float32")
    x2 = keras.layers.Input(shape = (299, 299, 3), name = "input_2", dtype = "float32")
    
    # define Xception Network
    xception = keras.applications.Xception(input_shape = (299, 299, 3), include_top = False, weights = "imagenet", pooling = 'avg')
    flat = keras.layers.Flatten(name = "flatten")
    dense1 = keras.layers.Dense(1024, activation = "elu", name = "dense1")
    dense2 = keras.layers.Dense(128, activation = "elu", name = "dense2")
    
    out1 = xception(x1)
    out1 = flat(out1)
    out1 = dense1(out1)
    out1 = dense2(out1)
    
    out2 = xception(x2)
    out2 = flat(out2)
    out2 = dense1(out2)
    out2 = dense2(out2)
    
    add = keras.layers.Add(name = "addition")([out1, out2])
    fc1 = keras.layers.Dense(128, activation = "elu", name = "fc1")(add)
    y = keras.layers.Dense(2, activation = "softmax", name = "output")(fc1)
    
    return keras.Model(inputs = [x1, x2], outputs = y)