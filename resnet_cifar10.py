
# coding: utf-8

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, MaxPooling2D, Input, Activation
from keras.layers import Add, Flatten, AveragePooling2D, Dense, BatchNormalization
from keras.models import Model
from keras import optimizers 

# n = 3
# Block 1
main_input = Input(shape=(32,32,3))
L1 = Conv2D(16, (3, 3), padding='same')(main_input)
L1 = BatchNormalization()(L1)
L1 = Activation('relu')(L1)
L2 = Conv2D(16, (3, 3), padding='same')(L1)
L2 = BatchNormalization()(L2)
L2 = Activation('relu')(L2)
L2 = Conv2D(16, (3, 3), padding='same')(L2)
L2 = BatchNormalization()(L2)
L2 = Activation('relu')(L2)
L2 = Conv2D(16, (3, 3), padding='same')(L2)
L2 = BatchNormalization()(L2)
L2 = Activation('relu')(L2)
L2 = Conv2D(16, (3, 3), padding='same')(L2)
L2 = BatchNormalization()(L2)
L2 = Activation('relu')(L2)
L2 = Conv2D(16, (3, 3), padding='same')(L2)
L2 = BatchNormalization()(L2)
L2 = Activation('relu')(L2)
L3 = Add()([L2, L1])

#Block 2
L3 = Conv2D(32, (3, 3), strides=(2,2), padding='same')(L3)
L3 = BatchNormalization()(L3)
L3 = Activation('relu')(L3)
L4 = Conv2D(32, (3, 3), padding='same')(L3)
L4 = BatchNormalization()(L4)
L4 = Activation('relu')(L4)
L4 = Conv2D(32, (3, 3), padding='same')(L4)
L4 = BatchNormalization()(L4)
L4 = Activation('relu')(L4)
L4 = Conv2D(32, (3, 3), padding='same')(L4)
L4 = BatchNormalization()(L4)
L4 = Activation('relu')(L4)
L4 = Conv2D(32, (3, 3), padding='same')(L4)
L4 = BatchNormalization()(L4)
L4 = Activation('relu')(L4)
L4 = Conv2D(32, (3, 3), padding='same')(L4)
L4 = BatchNormalization()(L4)
L4 = Activation('relu')(L4)
L5 = Add()([L4, L3])

# Block 3
L5 = Conv2D(64, (3, 3), strides=(2,2), padding='same')(L5)
L5 = BatchNormalization()(L5)
L5 = Activation('relu')(L5)
L6 = Conv2D(64, (3, 3), padding='same')(L5)
L6 = BatchNormalization()(L6)
L6 = Activation('relu')(L6)
L6 = Conv2D(64, (3, 3), padding='same')(L6)
L6 = BatchNormalization()(L6)
L6 = Activation('relu')(L6)
L6 = Conv2D(64, (3, 3), padding='same')(L6)
L6 = BatchNormalization()(L6)
L6 = Activation('relu')(L6)
L6 = Conv2D(64, (3, 3), padding='same')(L6)
L6 = BatchNormalization()(L6)
L6 = Activation('relu')(L6)
L6 = Conv2D(64, (3, 3), padding='same')(L6)
L6 = BatchNormalization()(L6)
L6 = Activation('relu')(L6)
L7 = Add()([L6, L5])

L8 = AveragePooling2D(pool_size=(2,2))(L7)
L8 = Flatten()(L8)
L8 = Dense(10,activation='softmax')(L8)
model = Model(main_input, L8)
model.summary()


(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# Normalize the pixel values
x_train = x_train.astype('float32')/255 
x_test = x_test.astype('float32')/255 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

# define the optimizer to be used
sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True) 
model.compile(optimizer=sgd, 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
history = model.fit(x_train, y_train, epochs=5, shuffle='batch', batch_size=64) 

test_loss, test_acc = model.evaluate(x_test, y_test) 
print(test_loss, test_acc)
f = open("log_resnet_cifar10_32x32.txt", 'w')
f.write("Test loss {}".format(test_loss))
f.write("\n")
f.write("Test accuracy {}".format(test_acc))
f.close()
