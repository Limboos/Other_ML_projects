from keras.callbacks import CSVLogger
from datetime import datetime
from matplotlib import pyplot as plt
import requests
from keras import backend as K
import os
import multiprocessing
import tqdm
import pickle
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.models import Model
from keras.applications import vgg16
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Bidirectional, LSTM, TimeDistributed, Dropout, \
    Reshape, Average, Conv1D, MaxPooling1D, concatenate, Conv2D, Activation, Dropout, BatchNormalization, MaxPooling2D, \
    GlobalAveragePooling1D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, optimizers, Sequential
import datetime


class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)

        true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))

        self.cat_true_positives.assign_add(true_poss)

    def result(self):
        return self.cat_true_positives


###datagen.flow_from_directory
batch = 32
directoryTrain = r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\Datasets\Dataset_3"
directoryValid = r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\Datasets\Dataset_4"

BATCH_SIZE = batch
NUM_CLASSES = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
traingen = gen.flow_from_directory(directoryTrain,
                                   batch_size=BATCH_SIZE,
                                   class_mode="categorical",
                                   shuffle=True,
                                   seed=42)
validgen = gen.flow_from_directory(directoryValid,
                                   batch_size=BATCH_SIZE,
                                   class_mode="categorical",
                                   shuffle=True,
                                   seed=42)

print(traingen.image_shape)
print(validgen.image_shape)
print(tf.__version__)
print(keras.__version__)
METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='acc'),
  CategoricalTruePositives(NUM_CLASSES, BATCH_SIZE),
]
input_K = keras.Input(shape=(224, 224, 3))
RES_MODEL = keras.applications.ResNet50(include_top=False,
                                        weights="imagenet",
                                        input_tensor=input_K)
for layer in RES_MODEL.layers[:143]:
    layer.trainable = False

# for i, layer in enumerate(RES_MODEL.layers):
#     print(i, layer.name, "-", layer.trainable)

to_res = (224, 224)

model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(RES_MODEL)
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(3, activation='softmax'))

earlystopper = EarlyStopping(monitor="acc", patience=7, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="acc", factor=0.5, patience=5, verbose=1, mode='max', min_lr=0.000001)
logdir = os.path.join(r"C:\Users\Student240914\Desktop\PycharmProjects\keylogger\Praca\log",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
checkpointer = ModelCheckpoint(logdir + '/best_model.h5'
                               , monitor="acc"
                               , verbose=1
                               , save_best_only=True
                               , save_weights_only=False)
model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=METRICS)  #

fit_history = model.fit_generator(generator=traingen, validation_data=validgen, validation_steps=2880 / batch,
                                 epochs=10, callbacks=[tensorboard_callback, checkpointer])
