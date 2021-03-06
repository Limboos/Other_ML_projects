import numpy as np
import tensorflow

import matplotlib.pyplot as plt
from tensorflow import keras
print(tensorflow.__version__)
print(keras.__version__)
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test,y_test) = cifar10.load_data()
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tensorflow.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


model = Sequential([
   Convolution2D(filters=128, kernel_size=(5,5), input_shape=(32,32,3), activation='relu', padding='same'),
   BatchNormalization(),
   Convolution2D(filters=128, kernel_size=(5,5), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPool2D((2,2)),
   Convolution2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'),
   BatchNormalization(),
   Convolution2D(filters=64, kernel_size=(5,5), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPool2D((2,2)),
   Convolution2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'),
   BatchNormalization(),
   Convolution2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'),
   BatchNormalization(),
   MaxPool2D((2,2)),
   Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
   BatchNormalization(),
   Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'),
   BatchNormalization(),
   Flatten(),
   Dense(units=32, activation="relu"),
   Dropout(0.15),
   Dense(units=16, activation="relu"),
   Dropout(0.05),
   Dense(units=10, activation="softmax")
])
optim = RMSprop(lr=0.001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])


#model = keras.models.load_model(r'C:\Users\Student240914\Desktop\PycharmProjects\keylogger\Praca')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip = False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale = 1. / 255,
    shear_range=0.05,
    zoom_range=0.05,
)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

batch_size = 32
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
datagen_valid = ImageDataGenerator(
   rescale = 1. / 255,)

x_valid = x_train[:100*batch_size]
y_valid = y_train[:100*batch_size]

valid_steps = x_valid.shape[0] // batch_size
validation_generator = datagen_valid.flow(x_valid, y_valid, batch_size=batch_size)

history = model.fit(
   train_generator,
   steps_per_epoch=len(x_train) // batch_size,
   epochs=50,
   validation_data=validation_generator,
   validation_freq=1,
   validation_steps=valid_steps,
   verbose=2
)
model.save(r'C:\Users\Student240914\Desktop\PycharmProjects\keylogger\Praca\Model_cifar_10')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

x_final_test = x_test / 255.0
eval = model.evaluate(x_final_test, y_test)