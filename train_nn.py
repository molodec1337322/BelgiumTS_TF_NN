import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os


def show_image(index=0):
    """ выводит изображение из тренировочного датасета """

    plt.figure()
    plt.imshow(train_images[index])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def load_data(data_directory):
    """ Загружает готовые датасеты """

    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


ROOT_PATH = os.getcdkw()
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

train_images, train_labels = load_data(train_data_directory)

show_image()

""" это вроде должно рабоать """