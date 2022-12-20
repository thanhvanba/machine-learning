import streamlit as st
import random_number as rn
import CNN_train as cnn
from PIL import ImageTk, Image
import draw_number as dn
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import io
from streamlit_option_menu import option_menu
import Show_ipynb as ip
model = tf.keras.models.load_model('./train/')
with st.sidebar:
    add_selectbox = option_menu("Main Menu", ["Home", 'KNN-digit-recognize','CNN-digit-recognize','Chart-data'], 
        icons=['house' ,'calculator','calculator','bar-chart'], menu_icon="cast", default_index=0)
    #add_selectbox
if 'key' not in st.session_state:
    st.session_state.key = 0
if add_selectbox=="Home":
  st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>NHẬN DẠNG CHỮ SỐ</h1>", unsafe_allow_html=True)
  st.image("https://www.mathedup.co.uk/wp-content/uploads/2015/08/pay-819587_1920.jpg")
  st.text('''Thành viên nhóm thực hiện:
                  1️⃣ Nguyễn Thanh Sang   20110710
                  2️⃣ Lê Anh Kiệt         20110
                  3️⃣ Văn Bá Trung Thành  20110722
            Đề tài : Nhận dạng chữ số bằng thuật toán KNN và CNN
            Link source code: github
            Link app đã deploy: www.''')
elif add_selectbox=="KNN-digit-recognize":
  st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>NHẬN DẠNG CHỮ SỐ</h1>", unsafe_allow_html=True)
  KNN_prediction_selected = option_menu(
        menu_title = 'KNN-digit-recognize',
        options = ['Drawing',"Upload",'Random'],
        orientation = "horizontal",
        menu_icon='house',
        icons=['pencil','upload','question-diamond']
        #display_toobar="reset"
    )
  #tab1, tab2, tab3 = st.tabs(["Drawing", "Upload", "Random"])


  if KNN_prediction_selected=='Drawing':
    st.header("Drawing Number - ✏️")
    dn.draw()
    btn_predict=st.button("Predict")
    if btn_predict:
      rn.btn_recognition_click() 

  if KNN_prediction_selected=='Upload':
    st.header("Upload Number - ⬆️")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
      image=Image.open(uploaded_file)
      st.image(image)
      image=image.save('digit.jpg')
    btn_pre=st.button("Predict Number")
    if btn_pre:
      rn.btn_recognition_click()  

  if KNN_prediction_selected=='Random':
    st.header("Random Number - 🔃")
    btn_create=st.button("Create Number")
    btn_reg=st.empty()
    #btn_rec=None
    if btn_create or st.session_state.key == 1:
      rn.btn_create_click()
      rn.resized()
      image=Image.open('digit.jpg')
      st.image(image)
      btn_reg=st.button("Predict Number")
    if btn_reg or st.session_state.key == 2:
      rn.btn_recognition_click()
#CNN
elif add_selectbox=="CNN-digit-recognize":
  st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>NHẬN DẠNG CHỮ SỐ</h1>", unsafe_allow_html=True)
  CNN_prediction_selected = option_menu(
        menu_title = 'CNN-digit-recognize',
        options = ['Drawing',"Upload",'Random'],
        orientation = "horizontal",
        menu_icon='house',
        icons=['pencil','upload','question-diamond']
    )
  #tab1, tab2, tab3 = st.tabs(["Drawing", "Upload", "Random"])


  if CNN_prediction_selected=='Drawing':
    st.header("Drawing Number - ✏️")
    dn.draw()
    btn_predict=st.button("Predict")
    if btn_predict:
      cnn.btn_recognition_click() 

  if CNN_prediction_selected=='Upload':
    st.header("Upload Number - ⬆️")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
      image=Image.open(uploaded_file)
      st.image(image)
      image=image.save('digit.jpg')
    btn_pre=st.button("Predict Number")
    if btn_pre:
      cnn.btn_recognition_click()  

  if CNN_prediction_selected=='Random':
    st.header("Random Number - 🔃")
    btn_create=st.button("Create Number")
    btn_rec=st.empty()
    if btn_create:
      cnn.btn_create_click()
      rn.resized()
      image=Image.open('digit.jpg')
      #rn.resized()
      st.image(image)
      btn_rec=st.button('Predict Number')
    if btn_rec:
      cnn.btn_recognition_click()  
else:
  st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>NHẬN DẠNG CHỮ SỐ</h1>", unsafe_allow_html=True)
  show_colab = option_menu(
        menu_title = 'CNN-digit-recognize',
        options = ['KNN',"CNN"],
        orientation = "horizontal",
        menu_icon='house',
        icons=['paper','paper']
    )
  if show_colab=='KNN':
    ip.display_KNN()
  else:
    ip.display_CNN()
     
  #if tab
  # tab



