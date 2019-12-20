import research_datasets as rd
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os


model = keras.models.load_model('model.h5')

img_path = os.path.join(os.getcwd(), 'Testing', '00000', '00017_00000.ppm')
img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
plt.imshow(img, cmap='gray')
plt.show()

x = image.img_to_array(img)
x = 255 - x
x /= 255
x = x.reshape(1, 28, 28)

predictions = model.predict(x)

