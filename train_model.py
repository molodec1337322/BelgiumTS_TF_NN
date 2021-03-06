from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import research_datasets as rd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
from keras.preprocessing import image


train_images, train_labels = rd.get_train_data()
test_images, test_labels = rd.get_test_data()

train_images_num = len(train_images)
test_images_num = len(test_images)

classes = []
classes = [sign_number for sign_number in test_labels if sign_number not in classes]
num_classes = len(classes)

print('Данные подготовлены')

hidden_size = 512
num_epochs = 80

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(hidden_size, activation=tf.nn.relu),
    keras.layers.Dense(hidden_size, activation=tf.nn.relu),
    keras.layers.Dense(num_epochs, activation=tf.nn.softmax)
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, nb_epoch=num_epochs, verbose=1, validation_split=0.1)

model.save('model.h5')
print('Модель сохранена!')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('\nТочность на проверочных данных:', test_acc)





