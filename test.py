import os
from model import SRCNN
from utils import load_test
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=None)
parser.add_argument('--label_size', type=int, default=None)
parser.add_argument('--c_dim', type=int, default=1)
parser.add_argument('--scale', type=int, default=3)

def main(args):
    srcnn = SRCNN(
        image_size=args.image_size,
        c_dim=args.c_dim,
        is_training=False)
    X_test, Y_test = load_test(scale=args.scale)
    predicted_list = []
    for img in X_test:
        predicted = srcnn.process(img.reshape(1,img.shape[0],img.shape[1],1))
        predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],1))
    n_img = len(predicted_list)
    dirname = './result'
    for i in range(n_img):
        imgname = 'image{:02}'.format(i)
        cv2.imwrite(os.path.join(dirname,imgname+'_input.bmp'), X_test[i])
        cv2.imwrite(os.path.join(dirname,imgname+'_answer.bmp'), Y_test[i])
        cv2.imwrite(os.path.join(dirname,imgname+'_predicted.bmp'), predicted_list[i])

if __name__ == '__main__':
    main(args=parser.parse_args())