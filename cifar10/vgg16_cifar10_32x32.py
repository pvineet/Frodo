# Training a modified VGG16 using CIFAR10 images

import numpy as np
import time
from keras import optimizers, layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import Adam, SGD

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after specfied epochs.
    Called automatically every epoch as part of callbacks during training.
     Arguments
        epoch (int): The number of epochs
     Returns
        lr (float32): learning rate
    """
    lr = 1e-2
    if epoch > 240:
        lr *= 0.5e-3
    elif epoch > 220:
        lr *= 1e-3
    elif epoch > 180:
        lr *= 1e-2
    elif epoch > 140:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Set width and heigt of the input
w = 32
h = 32

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Open log file to dump progress data
batch_size = 32
nb_epoch = 300

# Define a learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('vgg16_cifar10.csv')

# Instantiate a sequential model
model = Sequential()

# Block 1
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 2
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 3
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, (1, 1), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 4
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (1, 1), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 5
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(512, (1, 1), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# Flatten
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))

# The last Fully connected layers contribute the least hence removing it
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
summary =model.summary()

# scale pixel values from 0to255 => 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# convert labels from int to logits/one hot bits
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# declare a stochastic gradient descent optimizer with learning rate = 0.01 
# learning rate decays with 5e-5
sgd = optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9, nesterov=True)
# complile the model
# model.compile(optimizer=Adam(lr=lr_schedule(0)),
model.compile(optimizer=SGD(lr=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# train the model
start_time = time.time()
datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images
datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=nb_epoch,
                    verbose=1,
                    callbacks=[lr_reducer, lr_scheduler, csv_logger])

end_time = time.time()
f = open("vgg16_cifar10.log", "w")
print("Training time={}".format(end_time-start_time))

# evaluate the model on test images
test_loss, test_acc = model.evaluate(x_test, y_test)
f.write("Test Accuracy =", test_acc)
f.write("\nTest Loss =", test_loss)
f.write("\nTraining time = ", end_time-start_time)
f.close()
