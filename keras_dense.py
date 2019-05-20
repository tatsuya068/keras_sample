
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils


(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train =np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


model = Sequential()
model.add(Dense(512,activation = 'relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train,y_train,
                    batch_size=128,
                    epochs=20,
                    verbose=1,
                    validation_data = (x_test,y_test))
score = model.evaluate(x_test,y_test,verbose=0)
print('test loss -> ',score[0])
print('test accuracy -> ',score[1])




