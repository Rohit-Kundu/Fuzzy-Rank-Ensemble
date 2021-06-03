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

from cnn_utils import *
from ensemble_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = '.', help='Directory where the image data is stored')
parser.add_argument('--epochs', type=int, default = 20, help='Number of Epochs of training')
args = parser.parse_args()

path1 = args.data_directory
num_epochs = args.epochs

IMG_WIDTH=128
IMG_HEIGHT=128
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT,3)

df = createFrame(path1,IMG_DIM)
df = kFold(df)

target_names = os.listdir(path1)
num_classes = len(target_names)

for i in range(1,5):
  print(f"----------------------------------------------------FOLD NO {i}-------------------------------------------------------")
  dfTrain = df[df['kfold']!=i]
  dfTest = df[(df['kfold']==i)] 
  train_imgs = list(dfTrain[0])
  train_imgs = np.array(train_imgs)
  train_imgs = train_imgs/255
  train_labels = np.array(dfTrain[1])
  encoder = LabelEncoder()
  encoder.fit(train_labels)
  train_labels = encoder.transform(train_labels)
  train_labels = np_utils.to_categorical(train_labels)

  test_imgs = list(dfTest[0])
  test_imgs = np.array(test_imgs)
  test_imgs = test_imgs/255
  test_labels = np.array(dfTest[1])
  encoder = LabelEncoder()
  encoder.fit(test_labels)
  test_labels = encoder.transform(test_labels)
  test_labels = np_utils.to_categorical(test_labels)

  model0 = DenseNet(train_imgs,train_labels,class_no=num_classes,num_epochs=num_epochs)
  model1 = Inception(train_imgs,train_labels,class_no=num_classes,num_epochs=num_epochs)
  model2 = Xception(train_imgs,train_labels,class_no=num_classes,num_epochs=num_epochs)
  print("BASE LEARNERS ACCURACY-----------1.DENSENET 2.INCEPTION 3.XCEPTION")
  model0.evaluate(test_imgs, test_labels, batch_size=32)
  model1.evaluate(test_imgs, test_labels, batch_size=32)
  model2.evaluate(test_imgs, test_labels, batch_size=32)

  res1 = model1.predict(test_imgs)
  res2 = model0.predict(test_imgs) 
  res3 = model2.predict(test_imgs)
  predictedClass = doFusion(res1,res2,res3,test_labels,class_no=num_classes)

  leb1 = np.argmax(res1,axis=-1)
  leb2 = np.argmax(res2,axis=-1)
  leb3 = np.argmax(res3,axis=-1)
  actual = np.argmax(test_labels,axis=-1)
  
  print('Densenet-169 base learner')
  print(classification_report(actual, leb1,target_names = target_names,digits=4))
  print('Inception base learner')
  print(classification_report(actual, leb2,target_names = target_names,digits=4))
  print('Xception base learner')
  print(classification_report(actual, leb3,target_names = target_names,digits=4))
  
  print('Ensembled')
  print(classification_report(actual, predictedClass,target_names = target_names,digits=4))


  print(f"--------------------------------------------------END OF FOLD NO {i}--------------------------------------------------------")

