
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras import optimizers
import numpy as np
import h5py
import matplotlib.pyplot as plt


# In[2]:


#Load data
hf = h5py.File('cifar10_scaled_2.h5', 'r')
#train_images = hf.get('train_data')
#train_labels = hf.get('train_labels')
#test_images = hf.get('test_data')
#test_labels = hf.get('test_labels')
train_images = hf['train_data'][:]
test_images = hf['test_data'][:]
train_labels = hf['train_labels'][:]
test_labels = hf['test_labels'][:]
hf.close()
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255


# In[3]:


plt.imshow(train_images[0])


# In[4]:


plt.imshow(train_images[5])


# In[5]:


model = VGG16(include_top=True,weights=None,input_shape=(64, 64, 3), classes=10)
model.summary()


# In[6]:


y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, y_train, epochs=5, batch_size=64)


# In[ ]:


hf.close()


# In[ ]:


print(history)

