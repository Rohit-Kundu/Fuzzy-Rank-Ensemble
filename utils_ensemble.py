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


#Fuzzy Rank-based Ensemble:
def getScore(model,test_imgs):
  res = model.predict(test_imgs)
  return res 

def generateRank1(score,class_no):
  rank = np.zeros([class_no,1])
  scores = np.zeros([class_no,1])
  scores = score
  for i in range(class_no):
      rank[i] = 1 - np.exp(-((scores[i]-1)**2)/2.0)
  return rank

def generateRank2(score,class_no):
  rank = np.zeros([class_no,1])
  scores = np.zeros([class_no,1])
  scores = score
  for i in range(class_no):
      rank[i] = 1 - np.tanh(((scores[i]-1)**2)/2)
  return rank

def doFusion(res1,res2,res3,label,class_no):
  cnt = 0
  id = []
  for i in range(len(res1)):
      rank1 = generateRank1(res1[i],class_no)*generateRank2(res1[i],class_no)
      rank2 = generateRank1(res2[i],class_no)*generateRank2(res2[i],class_no)
      rank3 = generateRank1(res3[i],class_no)*generateRank2(res3[i],class_no)
      rankSum = rank1 + rank2 + rank3
      rankSum = np.array(rankSum)
      scoreSum = 1 - (res1[i] + res2[i] + res3[i])/3
      scoreSum = np.array(scoreSum)
      
      fusedScore = (rankSum.T)*scoreSum
      cls = np.argmin(rankSum)
      if cls<class_no and label[i][cls]== 1:
          cnt += 1
      id.append(cls)
  print(cnt/len(res1))
  return id
