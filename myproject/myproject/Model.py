from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import cv2
from django.conf import settings

class MyModel:
    

    def predict(img_path):
        IMAGE_SIZE = 148
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE,IMAGE_SIZE,1), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.load_weights('./'+settings.STATIC_URL+'weights.h5')
        i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        i = cv2.resize(i, (IMAGE_SIZE, IMAGE_SIZE))
        i = np.array(i).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        i = np.array(i).astype('float')/255.0
        prediction = model.predict([i])
        return (['NORMAL', 'PNEUMONIA'][int(prediction.round())], prediction[0][0])

