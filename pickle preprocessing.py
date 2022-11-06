# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:02:17 2022

@author:hp

"""

import pickle
import os
import numpy as np

import cv2
from tqdm import tqdm

#data preprocessing

CATEGORIES = ['nonviolent', 'violent']
IMG_SIZE =80
CLASS_NUM = 2
BATCH_SIZE = 32

IMAGE_SHAPE = (80,80, 3)


    
DATADIR = 'C:/projecto/train'
for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img))

training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):  # iterate over each image 
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

print(len(training_data))

X_train =[]
y_train=[]


for categories, label in training_data:
    X_train.append(categories)
    y_train.append(label)



X_train=np.array(X_train)
y_train=np.array(y_train)
#print(X_train[1])
#print(len(y_train))
DATADIR = 'C:/projecto/test'
for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))

test_data=[]
def create_test_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):  # iterate over each image 
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                test_data.append([new_array,class_num])
            except Exception as e:
                pass
create_test_data()

print(len(test_data))

X_test =[]
y_test=[]


for categories, label in test_data:
    X_test.append(categories)
    y_test.append(label)
    
    
    
    
# convert train data to pickle 
pickle_out = open("X_train_dpvd.pickle","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open("y_train_dpvd.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


#convert test data to pickle 

pickle_out = open("X_test_dpvd.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out = open("y_test_dpvd.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()

