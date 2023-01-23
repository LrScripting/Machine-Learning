## Convolution Neural Network using the inception pretrained model and the horse-or-human dataset on Kraggle.


import urllib
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
from tensorflow.keras.applications.inception_v3 import InceptionV3

weightsUrl = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weightsFile = "inception_v3.h5"
urllib.request.urlretrieve(weightsUrl, weightsFile)
inceptionModel = InceptionV3(input_shape=(150, 150, 3),
 weights=None,
 include_top=False)
inceptionModel(weightsFile)

for l in inceptionModel:
    l.trainable = False

finalLayer = inceptionModel.get_layer('mixed7')
finalOutput = finalLayer.output

#accuracy callback 
class accuracyGoal(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.9999):
            print('\nReached 90% Accuracy, model stopped.')
            self.model.stop_training = True

##test custom image against network
def testCustomImage(imageName):
    path = os.path.join(os.getcwd(), 'horse-or-human', 'custom', imageName)
    img = image.load_img(path, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    testImage = np.vstack([x])
    tests = model.predict(testImage)
    print(tests[0])
    if tests[0] > 0.5:
        print(imageName + " is human!")
    else:
        print(imageName + " is A HORSE!")

#graph image from training data
def graphImage(trainingData):
    plt.figure()
    plt.imshow(trainingData[0])
    plt.grid(True)
    plt.show()

#initialize accuracy callback fn
accGoal = accuracyGoal()

# set directories for imagegenerator
trainingDir = os.path.join(os.getcwd(), 'horse-or-human', 'train')
valDir = os.path.join(os.getcwd(), 'horse-or-human', 'validation')

#gen imagegenerator so image is in right format for network
DataGen = ImageDataGenerator(rescale=1./255)
trainGen = DataGen.flow_from_directory(trainingDir, target_size=(300,300), class_mode='binary')
valGen = DataGen.flow_from_directory(valDir, target_size=(300, 300), class_mode='binary')

customLayers = tf.keras.layers.Flatten()(finalOutput)
customLayers = tf.keras.layers.Dense(1024, activation='relu')(customLayers)
customLayers = tf.keras.layers.Dense(1, activation='sigmoid')(customLayers)

model = tf.keras.models.Sequential([inceptionModel, customLayers
])


model.compile(loss='binary_crossentropy',
 optimizer=RMSprop(learning_rate=0.001),
 metrics=['accuracy'], )



history = model.fit(
 trainGen,
 epochs=7,
 validation_data=valGen,
 callbacks=accGoal
)

testCustomImage("horse1.jpeg")
testCustomImage("horse2.jpg")
testCustomImage("human.jpg")
