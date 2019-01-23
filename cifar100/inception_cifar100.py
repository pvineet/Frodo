import time
import numpy as np
from keras import layers
from keras import models
from keras.layers import Concatenate, Activation, BatchNormalization, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Input, AveragePooling2D, Dense
from keras.models import Model
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback
from keras.optimizers import Adam, SGD

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class Inception:
    def __init__(self, num_inception_modules=4, shape=(32, 32, 3)):
	main_input = Input(shape=shape)
	L1 = Conv2D(32, (3, 3), padding='same')(main_input)
	L1 = BatchNormalization()(L1)
	L1 = Activation('relu')(L1)
	L1 = Conv2D(32, (1, 1), padding='same')(L1)
	L1 = Activation('relu')(L1)
	L1 = Conv2D(32, (3, 3), padding='same')(L1)
	L1 = Activation('relu')(L1)
	L1 = BatchNormalization()(L1)
	# Inception Module
	for i in range(num_inception_modules):
        L1 = self.add_inception(L1)

	L2 = AveragePooling2D(pool_size=(2,2))(L1)
	L2 = Flatten()(L2)
	L2 = Dense(512)(L2)
	L2 = Dense(100,activation='softmax')(L2)
	self.model = Model(main_input, L2)
	self.model.summary()

    def add_inception(self, L1):
        B1 = Conv2D(128,(1,1), padding='same')(L1)
        B1 = Activation('relu')(B1)
        B2 = Conv2D(16,(1,1), padding='same')(L1)
        B2 = Activation('relu')(B2)
        B2 = Conv2D(32,(5,5), padding='same')(B2)
        B2 = Activation('relu')(B2)
        B3 = Conv2D(96,(1,1), padding='same')(L1)
        B3 = Activation('relu')(B3)
        B3 = Conv2D(128,(3,3), padding='same')(B3)
        B3 = Activation('relu')(B3)
        L2 = layers.concatenate([B1, B2, B3], axis=3)
        return L2

    def build(self):
	return self.model

    def callbacks(self):
        # define a learning rate scheduler
        callbacks = []
        callbacks.append(LearningRateScheduler(self.lr_schedule))
        #callbacks.append(ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6))
        #callbacks.append(EarlyStopping(min_delta=0.001, patience=10))
        callbacks.append(CSVLogger('vgg16_cifar100_1.csv'))
        callbacks.append(TimeHistory())
        return callbacks

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        Arguments
            epoch (int): The number of epochs
        Returns
            lr (float32): learning rate
        """
        lr = 1e-1
        if epoch > 240:
            lr *= 0.5e-3
        elif epoch > 220:
            lr *= 1e-3
        elif epoch > 160:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

batch_size = 32
nb_epoch = 300

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inception = Inception()
model = inception.build()

#sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True) 
model.compile(optimizer=SGD(lr=inception.lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images
datagen.fit(x_train)

#start_time = time.time()
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=nb_epoch,
                    verbose=1,
                callbacks=inception.callbacks())
#end_time = time.time()

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)

