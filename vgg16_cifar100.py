# Training a modified VGG16 using CIFAR10 images
import time
import numpy as np
from keras.layers import (
		Dense, 
		Dropout, 
		Activation, 
		Flatten,
		Input,
		Conv2D, 
		MaxPooling2D,
		BatchNormalization)
from keras.utils import to_categorical
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
		ReduceLROnPlateau, 
		CSVLogger, 
		EarlyStopping, 
		LearningRateScheduler,
		Callback)
from keras.optimizers import Adam, SGD
from keras.models import Model

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class vgg16:
    def __init__(self, fc_layer_size = 512, conv_dropout = False):
	input_layer = Input(shape=(32,32,3))
	#Block 1
	x = self.conv_layer(input_layer)
	#Block 2
	x = self.conv_layer(x, n=128)
	#Block 3
	x = self.conv_layer(x, n=256, conv_1_1=True)
	#Block 4
	x = self.conv_layer(x, n=512, conv_1_1=True)
	#Block 5
	x = self.conv_layer(x, n=512, conv_1_1=True)
	#Flatten
	x = Flatten()(x)
	for i in range(3):
	    x = Dense(512,activation='relu')(x)
	    if i < 2:
		x = Dropout(0.5)(x)
	x = Dense(100, activation='softmax')(x)
	self.model = Model(input_layer, x)
	summary = self.model.summary()
	
    def build(self):
	return self.model
	
    def conv_layer(self, x, n=64, conv_1_1=False, dropout_layer=False, batch_norm=True):
	"""
        n =  number of filters/channels
	"""
	x = Conv2D(n, (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(n, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
	if batch_norm:
	    x = BatchNormalization()(x)
 	if dropout_layer:
	    x = Dropout(0.5)(x)
	return x	

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
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
    
    def log_file(self):
	pass

    def callbacks(self,):
        # define a learning rate scheduler
        callbacks = []
        callbacks.append(LearningRateScheduler(self.lr_schedule))
        #callbacks.append(ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6))
        #callbacks.append(EarlyStopping(min_delta=0.001, patience=10))
        callbacks.append(CSVLogger('vgg16_cifar100.csv'))
        callbacks.append(TimeHistory())
        return callbacks

#Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
vgg16 = vgg16(fc_layer_size = 1024)
model = vgg16.build()

#open log file to dump progress data
batch_size = 32
nb_epoch = 300

#scale pixel values from 0to255 => 0 to 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# convert labels from int to logits/one hot bits
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#complile the model
#model.compile(optimizer=Adam(lr=lr_schedule(0)),
model.compile(optimizer=SGD(lr=vgg16.lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#  train the model
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
                    callbacks=vgg16.callbacks())
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}".format(test_loss))
print("Test Accuracy: {}".format(test_accuracy))
print(time_callback.times)
