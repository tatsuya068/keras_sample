import cv2
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D, Activation,Flatten
from keras.layers import Dropout
from keras.optimizers import RMSprop
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


data1_dir = '/- file dir -/'
data2_dir = '/- file dir -/'


files_1 = glob.glob(data1_dir+'/*.jpg')
files_2 = glob.glob(data2_dir+'/*.jpg')


X = []
Y = []

for file_name in files_1:
    img = cv2.imread(file_name)
    img = cv2.resize(img,(128,128))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X.append(img)
    Y.append(0)


for file_name in files_2:
    img = cv2.imread(file_name)
    img = cv2.resize(img,(128,128))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X.append(img)
    Y.append(1)


X = np.asarray(X)
Y = np.asarray(Y)
X = X.astype('float32')
X = X / 255.0

Y = np_utils.to_categorical(Y,2)

x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.30,
                                                    random_state=111,
                                                    shuffle=True)

print('x_train : ',x_train.shape)
print('x_test : ',x_test.shape)
print('y_train : ',y_train.shape)
print('y_test : ',y_test.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), 
          padding='same',
          input_shape=(128,128,3),
	  activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3),
          padding = 'same',
	  activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))     
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=5,
          epochs=20,
	  verbose=1,
	  validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

