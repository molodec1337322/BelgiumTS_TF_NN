from skimage import io
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import math
import os


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
            images.append(io.imread(f))
            labels.append(int(d))
    return images, labels


def color_to_gray(images):
    """ Преобразует изображения в оттенки серого """

    images_arr = np.array(images)
    return rgb2gray(images_arr)


def get_train_data():
    """ подготавливает данные для обучения и передачи их в НС """

    ROOT_PATH = os.getcwd()
    train_data_directory = os.path.join(ROOT_PATH, "Training")

    train_images, train_labels = load_data(train_data_directory)

    # трансформация изображений до размера 28х28 пикселей и их конвертация в серый цвет
    train_images_transformed = [transform.resize(image, (28, 28)) for image in train_images]
    train_images_transformed = color_to_gray(train_images_transformed)
    return train_images_transformed, train_labels


def get_test_data():
    """ подготавливает данные для тестирования НС """

    ROOT_PATH = os.getcwd()
    test_data_directory = os.path.join(ROOT_PATH, "Testing")

    test_images, test_labels = load_data(test_data_directory)

    # трансформация изображений до размера 28х28 пикселей и их конвертация в серый цвет
    test_images_transformed = [transform.resize(image, (28, 28)) for image in test_images]
    test_images_transformed = color_to_gray(test_images_transformed)
    return test_images_transformed, test_labels






