import research_datasets as rd
import tensorflow as tf
from tensorflow import keras

# инициализация плейсхолдеров
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# сглаживание входных данных
images_flat = tf.contrib.layers.flatten(x)

# Полностью подключенный слой
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# функция потерь
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# оптимизатор
train_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)
