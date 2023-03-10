import streamlit as st
from streamlit_drawable_canvas import st_canvas
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')

#AIモデルの定義
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
from tensorflow.keras.utils import to_categorical

x_train = x_train.reshape(60000,28,28,1)
x_test =x_test.reshape(10000,28,28,1)

x_train=x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Activation

model = Sequential()
model.add(Conv2D(16,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
model.add(Conv2D(16,(2,2),strides=2,padding='same'))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Conv2D(32,(2,2),strides=2,padding='same'))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(2,2),strides=2,padding='same'))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

#モデルの学習に必要な設定
from tensorflow.keras.optimizers import Adam
model.compile(Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

#学習
history = model.fit(x_train,y_train,batch_size=128,epochs=15,verbose=1,validation_data=(x_test,y_test))

#学習済みモデルの保存
model.save('my_model.h5')