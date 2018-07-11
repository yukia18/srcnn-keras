import os
import cv2
import numpy as np
from PIL import Image

def load_train(image_size=33, label_size=21, stride=21):
    dirname = './train'
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    images = [img[0:img.shape[0]-np.mod(img.shape[0],3),0:img.shape[1]-np.mod(img.shape[1],3)] for img in images]

    trains = images.copy()
    labels = images.copy()
    
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
    dirname = './test'
    dir_list = os.listdir(dirname)
    images = [cv2.cvtColor(cv2.imread(os.path.join(dirname,img)),cv2.COLOR_BGR2GRAY) for img in dir_list]
    images = [img[0:img.shape[0]-np.mod(img.shape[0],3),0:img.shape[1]-np.mod(img.shape[1],3)] for img in images]

    tests = images.copy()
    labels = images.copy()
    
    tests = [cv2.resize(img, None, fx=1/3., fy=1/3.) for img in tests]
    tests = [cv2.resize(img, None, fx=3/1., fy=3/1.) for img in tests]
    
    tests = [img.reshape(img.shape[0],img.shape[1],1) for img in tests]
    labels = [img.reshape(img.shape[0],img.shape[1],1) for img in labels]

    tests = np.array(tests)
    labels = np.array(labels)

    return tests, labels

def mse(y, t):
    return np.mean(np.square(y - t))

def psnr(y, t):
    return 20 * np.log10(255) - 10 * np.log10(mse(y, t))

def ssim(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov = np.mean((x - mu_x) * (y - mu_y))
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))



