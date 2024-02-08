#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)


# In[5]:


os.chdir('C:/Users/hardi/Python_Lab_projects/Project 5/brain_tumor')
if os.path.isdir('train/glioma_tumor') is False:
  os.mkdir('train')
  os.mkdir('valid')
  os.mkdir('test')

  for i in ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor' ]:
    shutil.move(f'{i}', 'train')
    os.mkdir(f'valid/{i}')
    os.mkdir(f'test/{i}')

    valid_samples = random.sample(os.listdir(f'train/{i}'),75)
    for j in valid_samples:
      shutil.move(f'train/{i}/{j}', f'valid/{i}')
    
    test_samples = random.sample(os.listdir(f'train/{i}'),50)
    for k in test_samples:
      shutil.move(f'train/{i}/{k}',f'test/{i}')

os.chdir('../..')


# In[6]:


train_path = 'C:/Users/hardi/Python_Lab_projects/Project 5/brain_tumor/train'
valid_path = 'C:/Users/hardi/Python_Lab_projects/Project 5/brain_tumor/valid'
test_path = 'C:/Users/hardi/Python_Lab_projects/Project 5/brain_tumor/test'


# In[7]:


train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)    .flow_from_directory(directory = train_path, target_size = (224, 224), batch_size = 10 )
valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)    .flow_from_directory(directory = valid_path, target_size = (224, 224), batch_size = 10 )
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)    .flow_from_directory(directory = test_path, target_size = (224, 224),  batch_size = 10, shuffle = False )


# In[8]:


imgs, labels = next(train_batches)


# In[9]:


def plotImages(images_arr):
  fig, axes = plt.subplots(1,10, figsize =(20,20))
  axes = axes.flatten()
  for img, ax in zip (images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()


# In[10]:


plotImages(imgs)
print(labels)


# In[11]:


model = Sequential([
        Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape =(224,224,3)),
        MaxPool2D(pool_size = (2,2), strides = 2),
        Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (2,2), strides =2),
        Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (2,2), strides =2),
        Flatten(),
        Dense(units = 4, activation ='softmax')   
])


# In[12]:


model.summary()


# In[13]:


model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[14]:


model.fit(x = train_batches, validation_data = valid_batches, epochs = 5, verbose =2)


# In[15]:


test_imgs, test_labels = next(test_batches)
plotImages (test_imgs)
print(test_labels)


# In[16]:


test_batches.classes


# In[17]:


predictions = model.predict(x=test_batches, verbose =0)


# In[18]:


np.round(predictions)


# In[19]:


def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion Matrix',
                          cmap = plt.cm.Blues):
  plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation = 45)
  plt.yticks(tick_marks, classes)
  if normalize:
    cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
    print('Normalized confusion matrix')
  else:
    print('Confusion matrix, without normalization')
  print(cm)
  thresh = cm.max()/2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i,j],
             horizontalalignment = 'center',
             color = 'white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[20]:


cm = confusion_matrix(y_true = test_batches.classes, y_pred = np.argmax(predictions, axis = -1))


# In[21]:


cm_plots_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
plot_confusion_matrix(cm = cm, classes = cm_plots_labels, title = 'Confusion Matrix')


# In[41]:


import os.path
if os.path.isfile("C:/Users/hardi/Python_Lab_projects/Project 5/y0103479_5_model.h5") is False:
    model.save("C:/Users/hardi/Python_Lab_projects/Project 5/y0103479_5_model.h5")

