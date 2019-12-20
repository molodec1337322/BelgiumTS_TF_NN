import research_datasets as rd
from tensorflow import keras


images, labels = rd.get_train_data()
test_images, test_labels = rd.get_test_data()

model = keras.models.load_model('model.h5')

predictions = model.predict(test_images)
print(predictions)
