import os
import os.path as path

import np as np
import tensorflow as tf
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from tensorflow import set_random_seed
os.environ['PYTHONHASHSEED'] = "0"
np.random.seed(1)
set_random_seed = tf.random.set_seed(2)
# Any results you write to the current directory are saved as output.

import keras
from pandas import DataFrame
import numpy as np
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras import Sequential
from tensorflow_core.python.tools import freeze_graph, optimize_for_inference_lib
#from tensorflow_core.python.training import saver

MODEL_NAME = 'rune_classifier'

scale_percent = 1  # percent of original size

import matplotlib.pyplot as plt

import PIL

from sklearn.metrics import roc_auc_score,roc_curve, recall_score, confusion_matrix

#%matplotlib inline
from keras_preprocessing.image import ImageDataGenerator




def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same",
                     input_shape=(64, 64, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.4))
    model.add(Dense(6, activation="softmax"))

    return model

def train(model):
    model.compile(Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # gen=ImageDataGenerator()

    train_batches = gen.flow_from_directory("training", model.input_shape[1:3], color_mode="grayscale", shuffle=True,
                                            seed=1,
                                            batch_size=16)
    valid_batches = gen.flow_from_directory("Validation", model.input_shape[1:3], color_mode="grayscale", shuffle=True,
                                            seed=1,
                                            batch_size=16)
    test_batches = gen.flow_from_directory("Testing", model.input_shape[1:3], shuffle=False,
                                           color_mode="grayscale", batch_size=8)

   # model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=163, epochs=5, validation_data=valid_batches,
                                   validation_steps=624)

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out',MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, False, 'out/' + MODEL_NAME + '.chkp', output_node_name,"save/restore_all", "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")



def main():
    if not path.exists('out'):
        os.mkdir('out')


    model = build_model()

    train(model)

    export_model(tf.train.Saver, model, ["conv2d_1_input"], "dense_2/Softmax")


if __name__ == '__main__':
    main()








