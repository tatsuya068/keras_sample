import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils



# load the mnist data

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# check the data shape
print('x_train = '+str(x_train.shape))
print('y_train = '+str(y_train.shape))
print('x_test = '+str(x_test.shape))
print('x_test = '+str(y_test.shape))


# reshape the data to 28x28 pix
x_train = x_train.reshape(60000,28,28,1)
x_test  = x_test.reshape(10000,28,28,1)

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# Normalization
x_train /= 255
x_test /= 255

# convert one-hot encording
y_train = np_utils.to_categorical(y_train, 10)
y_test  = np_utils.to_categorical(y_test,10)

# model architecture

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),
                 activation = 'relu',
                 input_shape = (28,28,1),
                 padding = 'same'))

model.add(Conv2D(64, kernel_size=(3,3),
                 activation = 'relu',
                 padding = 'same'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer = RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train,y_train,
                    batch_size=100,
                    epochs = 10,
                    verbose = 1,
                    validation_data = (x_test,y_test))

score = model.evaluate(x_test,y_test)

print('[loss] : ',score[0])
print('[accuracy] : ', score[1])



