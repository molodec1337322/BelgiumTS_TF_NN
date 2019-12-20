import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import math
import os


def show_image(index=0):
    """ выводит изображение из тренировочного датасета """

    plt.figure()
    plt.axis('off')
    plt.imshow(train_images[index])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_images(count=25, size_x=10, size_y=10, images=None):
    """ выводит группу изображений """

    # создание поля с указанным размером
    plt.figure(figsize=(size_x, size_y))
    for i in range(count):
        # указывается количество столюцов и строк
        plt.subplot(math.sqrt(count).__round__(), math.sqrt(count).__round__(), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    plt.show()


def show_distribution_signs():
    """ выводит распределение данных по меткам """

    plt.hist(train_labels, 62)
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
            images.append(io.imread(f))
            labels.append(int(d))
    return images, labels


def color_to_gray(images):
    """ Преобразует изображения в оттенки серого """

    images_arr = np.array(images)
    return rgb2gray(images_arr)


def get_data():
    """ подготавливает данные для передачи в НС """

    ROOT_PATH = os.getcwd()
    train_data_directory = os.path.join(ROOT_PATH, "Training")
    #test_data_directory = os.path.join(ROOT_PATH, "Testing")

    train_images, train_labels = load_data(train_data_directory)

    # трансформация изображений до размера 28х28 пикселей и их конвертация в серый цвет
    train_images_transformed = [transform.resize(image, (28, 28)) for image in train_images]
    train_images_transformed = color_to_gray(train_images_transformed)
    return train_images_transformed







