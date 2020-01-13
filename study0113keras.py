from keras.utils import plot_model
from keras.preprocessing.image import img_to_array
from PIL import Image

import os
import shutil

# root=os.getcwd()
# datapath=os.path.join(root,"data")

from keras.preprocessing.image import ImageDataGenerator

from keras import layers
from keras import models
from keras.utils import plot_model

import platform
import tensorflow
import keras

import numpy as np

from keras.models import load_model
# model=load_model('')
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# token_index = {}
# for sample in samples:
#     for word in sample.split():
#         if word not in token_index:
#             token_index[word]=len(token_index)+1
#
# max_length=10
# results=np.zeros((len(samples),max_length,max(token_index())+1))
#
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         results[i, j, index] = 1.

#查看model
root=os.getcwd()
modelpat=os.path.join(root,'model_data')
modelname="yolo_weights.h5"
modelfilepath=os.path.join(modelpat,modelname)
yolomodel=load_model(modelfilepath)
yolomodel.summary()
