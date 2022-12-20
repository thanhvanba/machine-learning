import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2
#from tensorflow.python.keras
from tensorflow.python.keras import backend as k
mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0


y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

from tensorflow.python.keras import layers


model = tf.keras.models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(128,activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.BatchNormalization())
#model.add(layers.Dense(64,activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(10,activation='softmax'))

model.build(input_shape=x_train.shape)

model.summary()
model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30,validation_data=(x_test, y_test))
model.summary()
model.save('dense')
#test_loss,test_acc=model.evaluate(x_)

