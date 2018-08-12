# Training a modified VGG16 using CIFAR10 images

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import optimizers
from keras import layers
from keras.datasets import cifar10
import numpy as np

#set wifth and heigt of the input
w = 32
h = 32
#set the number of epochs
epochs = 15

#Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#open log file to dump progress data
f = open('log_cifar10_32x32.txt','w')

#instantiate a sequential model
model = Sequential()
#Block 1
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
#Block 2
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
#Block 3
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
#Block 4
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
#Block 5
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#Flatten
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
#The last Fully connected layers contribute the least hence removing it
#model.add(Dense(512,activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
summary =model.summary()
f.write(str(summary))

# scale pixel values from 0to255 => 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# convert labels from int to logits/one hot bits
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# declare a stochastic gradient descent optimizer with learning rate = 0.01 
# learning rate decays with 5e-5
sgd = optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9, nesterov=True)
#complile the model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#  train the model
history = model.fit(x_train, y_train, epochs=5, shuffle='batch', batch_size=64)
# evaluate the model on test images
test_loss, test_acc = model.evaluate(x_test, y_test)
# Write the training loss and accuracy to log file
f.write('Test loss \n')
f.write(str(test_loss))
f.write('\n')
f.write('Test accuracy\n')
f.write(str(test_acc))
f.write('\n')
f.write(str(history.history))
f.close()
