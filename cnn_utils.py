import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import pickle
import os
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow import keras 
from tensorflow.keras import layers
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix

#Data Augmentation
def augment(path,IMG_DIM):
  
  datagen = ImageDataGenerator(rotation_range=40,width_shift_range=.2,height_shift_range=.2,shear_range=.2,zoom_range=.2,horizontal_flip=True,fill_mode='nearest')

  #to list all directories in a specific folder
  directories = os.listdir(path)

  files_path = []
  labels = []
  for i in range(len(directories)):
    ls = []
    curPath = path +'/' +directories[i] + '/*'
    ls = glob.glob(curPath)
    temp = []
    for img in ls:
      x = img_to_array(load_img(img,target_size = IMG_DIM))
      x = x.reshape((1,)+x.shape)
      temp.append(x)
    
    i = 0
    target = 800
    for batch in datagen.flow(temp,batch_size=4,save_to_dir=curPath[:-1],save_format='jpg'):
      i += 1
      if len(ls) + i*4>800:
        break

#Creating Frame
def createFrame(path,IMG_DIM):
  train_imgs = []
  labels = []
  #getting all folder name
  directories = os.listdir(path)
  for i in range(len(directories)):
    ls = []
    temp = []
    curPath = path +'/' +directories[i] + '/*'
    #getting all files name
    ls = glob.glob(curPath)
    for img in ls:
      x = img_to_array(load_img(img,target_size = IMG_DIM))
      temp.append(x)

    #print(len(ls))
    train_imgs  = train_imgs + temp
    label = []
    label = [i]*len(ls)
    labels += label

  df = pd.DataFrame(list(zip(train_imgs,labels)))
  df = df.sample(frac = 1) 
  return df

def kFold(df):
  
  df['kfold'] = -1
  df = df.reset_index(drop=True)
  y = df[1]
  kf = model_selection.StratifiedKFold(n_splits=5)
  for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold'] = f

  return df

#Customized CNN models
def DenseNet(train_imgs,train_labels,class_no,num_epochs=20):
  print("-------------------------------------DENSENET--------------------------------------------")
  input_shape_densenet = (128, 128, 3)
  densenet_model = keras.applications.DenseNet169(include_top=False,weights="imagenet",input_tensor=None,input_shape=input_shape_densenet,pooling=None)
  densenet_model.trainable = True
  for layer in densenet_model.layers:
    layer.trainable = False

  layer = keras.layers.Flatten()(densenet_model.output)
  layer = keras.layers.Dense(units=1024,activation='relu')(layer)
  layer = keras.layers.Dropout(0.2)(layer)
  layer = keras.layers.Dense(units=128,activation='relu')(layer)
  layer = keras.layers.Dense(units=class_no,activation='softmax')(layer)
  model = keras.models.Model(densenet_model.input, outputs=layer)
  model.compile(optimizer = keras.optimizers.RMSprop(learning_rate=2e-5),loss='categorical_crossentropy',metrics=['acc'])

  history = model.fit(train_imgs, train_labels, batch_size=32, epochs=num_epochs,verbose=1)
  print("------------------------------------------------------------------------------------------")
  return model

def Inception(train_imgs,train_labels,class_no,num_epochs=20):
  print("-------------------------------------INCEPTION-------------------------------------------")

  pre_trained_model2 = keras.applications.InceptionV3(input_shape = (128,128,3),include_top = False,weights='imagenet')
  for layer in pre_trained_model2.layers:

    layer.trainable = False
  x = keras.layers.Flatten()(pre_trained_model2.output)
  x = layers.Dense(1028,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(64,activation='relu')(x)
  x = layers.Dense(class_no,activation='softmax')(x)
  model3 = Model(pre_trained_model2.input,x)
  model3.compile(optimizer = RMSprop(learning_rate=2e-5),loss='categorical_crossentropy',metrics=['acc'])
  history = model3.fit(x=train_imgs,y=train_labels, epochs = num_epochs, batch_size = 32,verbose=0)
  print("-----------------------------------------------------------------------------------------")
  return model3

def Xception(train_imgs,train_labels,class_no,num_epochs=20):
  print("-------------------------------------XCEPTION---------------------------------------------")
  pre_trained_model = keras.applications.Xception(input_shape = (128,128,3), include_top=False,weights="imagenet")
  for layer in pre_trained_model.layers:
    layer.trainable = False
  x = keras.layers.Flatten()(pre_trained_model.output)
  x = layers.Dense(256,activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(32,activation='relu')(x)
  x = layers.Dense(class_no,activation='softmax')(x)
  model1 = Model(pre_trained_model.input,x)
  model1.compile(optimizer = RMSprop(learning_rate=2e-5),loss='categorical_crossentropy',metrics=['acc'])
  history = model1.fit(x=train_imgs,y=train_labels, epochs = num_epochs, batch_size = 32, verbose=0)
  print("------------------------------------------------------------------------------------------")
  return model1
