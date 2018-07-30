import glob
import os
import cv2
import numpy as np
import pickle
WIDTH = 224
HEIGHT = 224
test_data_set = ['00']
train_data_set = ['01','02','03','04']
inp_dir = 'medico_dataset/'

imgs_train = []
imgs_test = []

X_test = []
y_test = []
X_train = []
y_train = []

for folder in train_data_set:
    imgs_train += glob.glob(inp_dir+folder+'/*')
    fo = open(inp_dir+'labels_'+folder+'.txt')
    for line in fo:
        y_train.append(int(line))

for folder in test_data_set:
    imgs_test += glob.glob(inp_dir+folder+'/*')
    fo = open(inp_dir+'labels_'+folder+'.txt')
    for line in fo:
        y_test.append(int(line))

for imgfile in imgs_test:
    img = cv2.imread(imgfile)
    img = cv2.resize(img,(WIDTH,HEIGHT))
    X_test.append(img)

for imgfile in imgs_train:
    img = cv2.imread(imgfile)
    img = cv2.resize(img,(WIDTH,HEIGHT))
    X_train.append(img)

X_test = np.array(X_test)
y_test = np.array(y_test).reshape((len(y_test),1))
X_train = np.array(X_train)
y_train = np.array(y_train).reshape((len(y_train),1)) 
print('X_test_shape: ',X_test.shape)
print('y_test_shape: ',y_test.shape)
print('X_train_shape: ',X_train.shape)
print('y_train_shape: ',y_train.shape)

of = open('medico_dataset.pickle',"wb")
pickle.dump(((X_train,y_train),(X_test,y_test)),of)
of.close()