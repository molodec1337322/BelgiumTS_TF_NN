import research_datasets as rd
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os


model = keras.models.load_model('model.h5')

test_images, test_labels = rd.get_test_data()
plt.imshow(test_images[0], cmap='gray')
plt.show()

x = image.img_to_array(test_images[0])
x = 255 - x
x /= 255
x = x.reshape(1, 28, 28)

prediction = model.predict(x)
print("Модель говорит, что это: ", test_labels(np.argmax(prediction)),
      "На самом деле это: ", test_labels[0])

