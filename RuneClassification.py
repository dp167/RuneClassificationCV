import os



# Any results you write to the current directory are saved as output.

#import keras
from pandas import DataFrame
import numpy as np
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import Sequential


import matplotlib.pyplot as plt

import PIL

from sklearn.metrics import roc_auc_score,roc_curve, recall_score, confusion_matrix

#%matplotlib inline
from keras_preprocessing.image import ImageDataGenerator

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from tensorflow import set_random_seed
os.environ['PYTHONHASHSEED'] = "0"
np.random.seed(1)
set_random_seed = tf.random.set_seed(2)
class CNN_run:

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",
                     input_shape=(64,64,1)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.4))
    model.add(Dense(2, activation="softmax"))
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))


    model.compile(Adam(lr=0.001),loss="binary_crossentropy", metrics=["accuracy"])

    gen = ImageDataGenerator(rescale= 1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = True)
    #gen=ImageDataGenerator()

    train_batches = gen.flow_from_directory("Training_data",model.input_shape[1:3],color_mode="grayscale",shuffle=True,seed=1,
                                            batch_size=16)
    valid_batches = gen.flow_from_directory("Validation_data", model.input_shape[1:3],color_mode="grayscale", shuffle=True,seed=1,
                                            batch_size=16)
    test_batches = gen.flow_from_directory("Test_data", model.input_shape[1:3], shuffle=False,
                                           color_mode="grayscale", batch_size=8)


    history1= model.fit_generator(train_batches,steps_per_epoch = 163,epochs=30,validation_data=valid_batches,validation_steps=624)
    test_accu = model.evaluate_generator(test_batches, steps=624)
    print('The testing accuracy is:',test_accu[1]*100,'%')

  #  history1 = model.fit_generator(test_batches,validation_data=valid_batches,epochs=3)



    p = model.predict_generator(test_batches, verbose=True)

    pre = DataFrame(p)
    pre["filename"] = test_batches.filenames
    pre["label"] = (pre["filename"].str.contains("RuneClassification")).apply(int)
    pre['pre'] = (pre[1]>0.5).apply(int)


    recall_score(pre["label"],pre["pre"])

    roc_auc_score(pre["label"],pre[1])

    true_positive_rate,false_positive_rate,threshold = roc_curve(pre["label"],pre[1])
    roc = DataFrame([true_positive_rate, false_positive_rate]).T
    #roc.plot(x=0,y=1)


    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')
    plt.show()

    plt.plot(history1.history['val_loss'])
    plt.plot(history1.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.show()









CNN_detect = CNN_run()
