import os
import cv2
import numpy as np
from PIL import Image

def load_train(image_size=33, label_size=21, stride=21):
    dirname = './train'
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    
    # crop images
    images = [img[0:img.shape[0]-np.mod(img.shape[0],3),0:img.shape[1]-np.mod(img.shape[1],3)] for img in images]

    # copy images for train and label
    trains = images.copy()
    labels = images.copy()
    
    # make train images worse
    trains = [cv2.resize(img, None, fx=1/3., fy=1/3.) for img in trains]
    trains = [cv2.resize(img, None, fx=3/1., fy=3/1.) for img in trains]

    sub_trains = []
    sub_labels = []

    for train, label in zip(trains, labels):
        v, h = train.shape
        padding = abs(image_size - label_size) // 2
        for x in range(0,v-image_size+1,stride):
            for y in range(0,h-image_size+1,stride):
                sub_train = train[x:x+image_size,y:y+image_size]
                sub_label = label[x+padding:x+padding+label_size,y+padding:y+padding+label_size]
                sub_train = sub_train.reshape(image_size,image_size,1)
                sub_label = sub_label.reshape(label_size,label_size,1)
                sub_trains.append(sub_train)
                sub_labels.append(sub_label)
    
    sub_trains = np.array(sub_trains)
    sub_labels = np.array(sub_labels)
    return sub_trains, sub_labels

def load_test():
    '''
    image shape is 256 * 256
    '''
    dirname = './test'
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    
    # crop images
    images = [img[0:img.shape[0]-np.mod(img.shape[0],3),0:img.shape[1]-np.mod(img.shape[1],3)] for img in images]

    # copy images for train and label
    trains = images.copy()
    labels = images.copy()
    
    # make train images worse
    trains = [cv2.resize(img, None, fx=1/3., fy=1/3.) for img in trains]
    trains = [cv2.resize(img, None, fx=3/1., fy=3/1.) for img in trains]

    # reshape
    trains = [img.reshape(img.shape[0],img.shape[1],1) for img in trains]
    labels = [img.reshape(img.shape[0],img.shape[1],1) for img in labels]

    return trains, labels




