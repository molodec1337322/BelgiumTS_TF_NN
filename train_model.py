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

height, width, depth = 28, 28, 1
train_images_num = len(train_images)
test_images_num = len(test_images)

print(train_images.shape)
print(test_images.shape)


classes = []
classes = [sign_number for sign_number in test_labels if sign_number not in classes]
num_classes = len(classes)

'''
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)
'''

print('Данные подготовлены')

hidden_size = 512
num_epochs = 80
batch_size = 128

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(62, activation=tf.nn.softmax)
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





