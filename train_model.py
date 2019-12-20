import research_datasets as rd
import tensorflow as tf
from tensorflow import keras


images, labels = rd.get_train_data()
test_images, test_labels = rd.get_test_data()
print('Данные подготовлены')

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(62, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(images, labels, epochs=40)

model.save('model.h5')
print('Модель сохранена!')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nТочность на проверочных данных:', test_acc)


