from __future__ import absolute_import, division, print_function, unicode_literals
import research_datasets as rd
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os


model = keras.models.load_model('model.h5')
image_number = 0
test_images, test_labels = rd.get_test_data()

while(True):

    plt.imshow(test_images[image_number])
    plt.show()

    x = image.img_to_array(test_images[image_number])

    x = x.reshape(1, 32, 32)

    prediction = model.predict(x)
    print("Модель говорит, что это: ", test_labels[np.argmax(prediction)],
          "На самом деле это: ", test_labels[image_number])
    image_number += 10

