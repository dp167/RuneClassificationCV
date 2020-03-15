import os
import os.path as path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib, freeze_graph

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from tensorflow import set_random_seed
os.environ['PYTHONHASHSEED'] = "0"
np.random.seed(1)
#set_random_seed = tf.random.set_seed(2)
# Any results you write to the current directory are saved as output.
from PIL import Image
from skimage import transform
#import keras
#from pandas import DataFrame
#import numpy as np
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras import Sequential
from keras.models import model_from_json
#from vis.visualization import visualize_saliency
#from vis.utils import utils

#from tensorflow_core.python.tools import freeze_graph, optimize_for_inference_lib
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
                     input_shape=(256, 256, 1)))
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





    gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    train_batches = gen.flow_from_directory("runes/mutated", model.input_shape[1:3], color_mode="grayscale", shuffle=True,
                                            seed=1,
                                            batch_size=16)
    valid_batches = gen.flow_from_directory("runes/validation", model.input_shape[1:3], color_mode="grayscale", shuffle=True,
                                            seed=1,
                                            batch_size=16)
    test_batches = gen.flow_from_directory("runes/testing", model.input_shape[1:3], shuffle=False,
                                          color_mode="grayscale", batch_size=8)

    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    history1 = model.fit_generator(train_batches, steps_per_epoch=163, epochs=5, validation_data=valid_batches,
                                   validation_steps=624)


    p = model.predict_generator(test_batches, verbose=True)


   # recall_score(pre["label"], pre["pre"])

    #roc_auc_score(pre["label"], pre[1])

    #true_positive_rate, false_positive_rate, threshold = roc_curve(pre["label"], pre[1])
   # roc = DataFrame([true_positive_rate, false_positive_rate]).T
    # roc.plot(x=0,y=1)

    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')
    plt.show()

    plt.plot(history1.history['val_loss'])
    plt.plot(history1.history['loss'])
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.show()

    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights_only = False
    model.save_weights("keras_model.h5")


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
    
'''
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

'''

def output_rune_name(highest_prediction):
    if (0 == highest_prediction):
        print("Ansuz")

    if (1 == highest_prediction):
        print("Fehu")
    if (2 == highest_prediction):
        print("Kano")
    if (3 == highest_prediction):
        print("Raido")

    if (highest_prediction == 4):
        print("Thurisaz")

    if (5 == highest_prediction):
        print("Uruz")


def test_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("keras_model256.h5")
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])


    np_image = Image.open("runes/validation/Kano/mut521.jpg")
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 1))
    np_image = np.expand_dims(np_image, axis=0)
    image1 = np_image

    np_image = Image.open("runes/saving/Ansuz/1.jpg")
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 1))
    np_image = np.expand_dims(np_image, axis=0)
    image2 = np_image
    cv2.imwrite("output_img2.png",image2)

    np_image = Image.open("runes/mutated/Thurisaz/mut61.jpg")
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (256, 256, 1))
    np_image = np.expand_dims(np_image, axis=0)
    image3 = np_image

    prediction1 = model.predict(image1)
    print("should be Kano")
    highest_prediction1 = np.argmax(prediction1)
    print(prediction1[0][highest_prediction1])
    output_rune_name(highest_prediction1)

    print("Should be Ansuz")
    prediction2 = model.predict(image2)
    highest_prediction2 = np.argmax(prediction2)
    print(prediction2[0][highest_prediction2])
    output_rune_name(highest_prediction2)

    print("should be Thurisaz")
    prediction3 = model.predict(image3)
    highest_prediction3 = np.argmax(prediction3)
    print(prediction3[0][highest_prediction3])
    output_rune_name(highest_prediction3)


def main():
    if not path.exists('out'):
        os.mkdir('out')
    #build_model()
    test_model()
    #freeze_graph = freeze_session(K.get_session(),output_names =[out.op.name for out in model.outputs])
    #export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")


if __name__ == '__main__':
    main()







