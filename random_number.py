

import tkinter as tk
from PIL import ImageTk, Image

import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2
import joblib
import streamlit as st

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
knn = joblib.load("knn_mnist.pkl")
index = None
       #st.button('Create Number')
def resized():
  img = cv2.imread('digit.jpg', cv2.IMREAD_UNCHANGED)
  scale_percent = 1000
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  cv2.imwrite('digit.jpg', resized)

def btn_create_click():
  index = np.random.randint(0, 9999, 1)
  digit = np.zeros((28,28), np.uint8)
  # digit = X_test[index]
  # cv2.imwrite('digit.jpg', digit) 
  # cv2.resize('digit.jpg', (28,28))   
  for x in range(0, 1):
    for y in range(0, 1):
      digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index]
      cv2.imwrite('digit.jpg', digit)
       
def btn_recognition_click():
  sp = cv2.imread('digit.jpg', cv2.IMREAD_GRAYSCALE)
  sample = cv2.resize(sp, (28,28))
  RESHAPED = 784
  sample = sample.reshape(1, RESHAPED) 
  predicted = knn.predict(sample)
  ketqua = predicted[0]
  st.write('This is digit :', ketqua)


        
# if __name__ == "__main__":
    
#     if st.button('Create Number'):
#       btn_create_click()
#       resized()
#       img=Image.open("digit.jpg")
#       st.image(img)
#     if(st.button('Predict')):
#       btn_recognition_click()


