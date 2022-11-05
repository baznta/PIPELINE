#import libraries
from keras.applications import vgg16
from keras.models import Model
from tensorflow.keras.models import Model
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import keras
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from distutils.dir_util import copy_tree
from tensorflow.keras import preprocessing
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
import matplotlib.pyplot as plt



#load the data from directories


CATEGORIES = ['nonviolent', 'violent']
IMG_SIZE =80
CLASS_NUM = 2
BATCH_SIZE = 1

IMAGE_SHAPE = (80,80, 3)


    
DATADIR = '/a self taught programmer/crime/train'
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
#lenofimage = len(training_data)
X_train =[]
y_train=[]


for categories, label in training_data:
    X_train.append(categories)
    y_train.append(label)
    
    


#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
#print(X.shape)
#X= np.array(X).reshape(lenofimage,-1)

X_train=np.array(X_train)
y_train=np.array(y_train)
DATADIR = '/a self taught programmer/crime/test'
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
#lenofimage = len(training_data)
X_test =[]
y_test=[]


for categories, label in test_data:
    X_test.append(categories)
    y_test.append(label)


#data preprocessing
print(X_test[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
X = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(X.shape)
X_test= np.array(X).reshape(-1)

X_test=np.array(X_test)
y_test=np.array(y_test)

#print(y.shape)
#X=X/255



#print(y.shape)
#X=X/255



X_train=X_train/255
X_test=X_test/255

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Split into validation set and testing set
#X_valid, X_test, y_valid,y_test=train_test_split(X_test,y_test,test_size=0.5,random_state=42)

print("Training set: ",X_train.shape)
#print("Validation set: ",X_valid.shape) 
print("Testing set: ",X_test.shape)


# Transforming non numerical labels into numerical labels
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels 
encoder.fit(y_train)
Y_train = encoder.transform(y_train)

# encoding test labels 
encoder.fit(y_test)
Y_test = encoder.transform(y_test)

# Transforming non numerical labels into numerical labels
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels 
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train)
'''
# Scaling the Train and Test feature set 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# encoding test labels 
encoder.fit(Y_test)
Y_test = encoder.transform(Y_test)
'''

#Load pretrained model
input_shape= (80,80,3)
vgg = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=input_shape)

#freaze layers
vgg.trainable = False
for layer in vgg.layers:
    layer.trainable = False
#feature extraction    
train_feature_extractor = vgg.predict(X_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0],-1)
print('Input to Feature Extractor Shape : ',X_train.shape)
print('Output of Feature Extractor Shape : ',train_feature_extractor.shape)
print('Input to Machine Learning Algorithm Shape',train_features.shape)

# Dimension of Train and Test set 
print("Dimension of Train set",X_train.shape)
print("Dimension of Test set",X_test.shape,"\n")



from sklearn.svm import SVC  
clf = SVC(kernel='linear') 
  
# fitting x samples and y classes 
clf.fit(train_features, y_train)
clf.score
'''
# View the accuracy score
print('Best score for training data:', clf.best_score_,"\n") 

# View the best parameters for the model found using grid search
print('Best C:',clf.best_estimator_.C,"\n") 
print('Best Kernel:',clf.best_estimator_.kernel,"\n")
print('Best Gamma:',clf.best_estimator_.gamma,"\n")

final_model = clf.best_estimator_
'''
np.array(X_test).reshape(-1, 1)
Y_pred = clf.predict(X_test)
Y_pred_label = list(encoder.inverse_transform(Y_pred))


# Making the Confusion Matrix+ 

print(confusion_matrix(Y_test,Y_pred_label))
print("\n")
print(classification_report(Y_test,Y_pred_label))
'''
print("Training set score for SVM: %f" % final_model.score(X_train , Y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test, Y_test ))
'''
