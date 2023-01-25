#Convolutional neaural network with multi classification and the rock paper scissors dataset from kraggle.com


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.optimizers import RMSprop

trainDir = os.path.join(os.getcwd(), 'Rock-Paper-Scissors', 'train')
valDir = os.path.join(os.getcwd(), 'Rock-Paper-Scissors', 'validation')

trainDataGen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.1, height_shift_range=0.2, zoom_range=0.2, fill_mode='nearest', horizontal_flip=True)
trainGen = trainDataGen.flow_from_directory(trainDir, target_size=(150,150), class_mode='categorical')
valGen = trainDataGen.flow_from_directory(valDir, target_size=(150,150), class_mode='categorical')

RPSmodel = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(320, activation='relu'),
    tf.keras.layers.Dense(320, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

RPSmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = RPSmodel.fit(trainGen, epochs=25, validation_data=valGen, verbose=1)
