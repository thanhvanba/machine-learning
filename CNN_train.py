import cv2

import tensorflow as tf
import numpy as np
import streamlit as st
import model_mnist_CNN as cnn
import random_number as rd
model = tf.keras.models.load_model('./train/')


def btn_recognition_click():
  image = cv2.imread('digit.jpg')
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
  newimg=tf.keras.utils.normalize(resized, axis=1)
  newimg=np.array(newimg).reshape(-1,28,28,1)
  prediction=model.predict(newimg)
  st.write("this is:",np.argmax(prediction))
  
def btn_create_click():
  index = np.random.randint(0, 9999, 1)
  digit = np.zeros((28,28), np.uint8)  
  for x in range(0, 1):
    for y in range(0, 1):
      digit[x*28:(x+1)*28, y*28:(y+1)*28] = rd.X_test[index]
      cv2.imwrite('digit.jpg', digit)