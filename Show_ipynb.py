import streamlit as st
import random_number as rn
from PIL import ImageTk, Image
import cv2
def display_KNN():
  st.code('''import tensorflow as tf
  from tensorflow import keras 
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import classification_report
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  import joblib''')

  st.code('''mnist = keras.datasets.mnist 
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data() ''')

  st.code('X_train.shape')
  st.write('(60000, 28, 28)')

  st.code('X_test.shape')
  st.write('(10000, 28, 28)')

  st.code("""index = np.random.randint(0, 60000)
  print('Index: ', index)
  print('Training Set: ')
  print('Label:', Y_train[index])
  img = np.asarray(X_train[index]).squeeze()
  plt.imshow(img)
  plt.show()""")
  st.write('''Index:  5856
  Training Set: 
  Label: 5''')
  st.image('output.png')

  st.code('''RESHAPED = 784
  X_train = X_train.reshape(60000, RESHAPED)
  X_test = X_test.reshape(10000, RESHAPED) ''')

  st.code('''(trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train, test_size=0.1, random_state=84)
  model = KNeighborsClassifier()
  model.fit(trainData, trainLabels)
  joblib.dump(model, "knn_mnist.pkl")''')
  st.write('''['knn_mnist.pkl']''')

  st.code('''predicted = model.predict(valData)
  do_chinh_xac = accuracy_score(valLabels, predicted)
  print('Độ chính xác trên tập validation: %.0f%%' % (do_chinh_xac*100))''')
  st.write('Độ chính xác trên tập validation: 97%')

  st.code('''predicted = model.predict(X_test)
  do_chinh_xac = accuracy_score(Y_test, predicted)
  print('Độ chính xác trên tập test: %.0f%%' % (do_chinh_xac*100))''')
  st.write('Độ chính xác trên tập test: 97%')

  st.code('''predicted = model.predict(X_test[0:10])
  x_test = X_test.reshape(10000, 28, 28)
  for i in range (5):
      print('Index: ', i)
      print('Label:', Y_test[i]) 
      print('Predict: ', predicted[i])
      img = x_test[i]
      plt.imshow(img, cmap = 'gray')
      plt.show()
  print(classification_report(Y_test[0:10], predicted))
  plt.show()''')
  st.write('''Index:  0\n
  Label: 7\n
  Predict:  7''')
  st.image('output2.png')

  st.write('''Index:  1\n
  Label: 2\n
  Predict:  2''')
  st.image('output3.png')

  st.write('''Index:  2\n
  Label: 1\n
  Predict:  1''')
  st.image('output4.png')

  st.write('''Index:  0\n
  Label: 7\n
  Predict:  7''')
  st.image('output2.png')

  st.write('''Index:  3\n
  Label: 0\n
  Predict:  0''')
  st.image('output5.png')
  st.write('''   precision    recall  f1-score   support
  0       1.00      1.00      1.00         1
  1       1.00      1.00      1.00         2
  2       1.00      1.00      1.00         1
  4       1.00      1.00      1.00         2
  5       1.00      1.00      1.00         1
  7       1.00      1.00      1.00         1
  9       1.00      1.00      1.00         2''')
  st.write("""accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10""")


def display_CNN():
  st.code('''import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from sklearn.metrics import accuracy_score''')
  st.code('''IMG_SIZE = 28

  mnist=tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train=tf.keras.utils.normalize(x_train,axis=1)
  x_test=tf.keras.utils.normalize(x_test,axis=1)
  x_trainr=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
  x_testr=np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)''')

  st.code('x_train.shape')
  st.write('(60000, 28, 28)')

  st.code('x_trainr.shape')
  st.write('(60000, 28, 28, 1)')

  st.code('''model=Sequential()
  model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(64,(3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation('relu'))

  model.add(Dense(32))
  model.add(Activation('relu'))

  model.add(Dense(10))
  model.add(Activation('softmax'))

  model.summary()
  model.compile( loss = "sparse_categorical_crossentropy",optimizer = 'adam', metrics=['accuracy'])

  model.fit(x_trainr,y_train,epochs=6,validation_split=0.3)
  model.summary()
  model.save('train')''')


  st.code('''Output exceeds the size limit. Open the full output data in a text editor
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  conv2d (Conv2D)              (None, 26, 26, 64)        640       
  _________________________________________________________________
  activation (Activation)      (None, 26, 26, 64)        0         
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928     
  _________________________________________________________________
  activation_1 (Activation)    (None, 11, 11, 64)        0         
  _________________________________________________________________
  max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
  _________________________________________________________________
  flatten (Flatten)            (None, 1600)              0         
  _________________________________________________________________
  dense (Dense)                (None, 64)                102464    
  _________________________________________________________________
  activation_2 (Activation)    (None, 64)                0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 32)                2080      
  _________________________________________________________________
  activation_3 (Activation)    (None, 32)                0         
  ...
  Trainable params: 142,442
  Non-trainable params: 0
  _________________________________________________________________
  INFO:tensorflow:Assets written to: train\assets ''')

  st.code('''redicted = model.predict(x_testr[0:10])
  for i in range (5):
      print('Index: ', i)
      print('Label:', y_test[i]) 
      print('Predict: ', np.argmax(predicted[i]))
      img = x_test[i]
      plt.imshow(img, cmap = 'gray')
      plt.show()
  plt.show()''')

  st.write('''Index:  0\n
  Label: 7\n
  Predict:  7''')
  st.image('output2.png')

  st.write('''Index:  1\n
  Label: 2\n
  Predict:  2''')
  st.image('output3.png')

  st.write('''Index:  2\n
  Label: 1\n
  Predict:  1''')
  st.image('output4.png')

  st.write('''Index:  3\n
  Label: 0\n
  Predict:  0''')
  st.image('output5.png')

  st.write('''Index:  4\n
  Label: 4\n
  Predict:  4''')
  st.image('output9.png')

