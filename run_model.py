from __future__ import absolute_import, division, print_function, unicode_literals
import research_datasets as rd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_image(i, predictions_array, true_labels, images):

  predictions_array = predictions_array[i]
  true_label = true_labels[i]
  image = images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(image, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  print('\nМодель говорит что это: ', predicted_label)
  print('На самом деле это: ', true_label)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{}  {:2.0f}%   ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)


def show_prediction(index):
    """ Показывает и визуализирует результат """

    i = index
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.show()


model = keras.models.load_model('model.h5')
test_images, test_labels = rd.get_test_data()

print(test_labels)

sign_numbers = []
sign_numbers = [sign_number for sign_number in test_labels if sign_number not in sign_numbers]

predictions = model.predict(test_images)

while(True):
    show_prediction(random.randint(0, len(test_labels)))




