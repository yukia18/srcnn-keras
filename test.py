from model import SRCNN
from utils import load_test
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--label_size', type=int, default=244)
parser.add_argument('--c_dim', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1500)

def main(args):
    srcnn = SRCNN(
        image_size=args.image_size,
        label_size=args.label_size,
        c_dim=args.c_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        is_training=False)
    X_test, Y_test = load_test()
    predicted = srcnn.process(X_test)
    cv2.imwrite('input.bmp', X_test)
    cv2.imwrite('predicted.bmp', predicted)


if __name__ == '__main__':
    main(args=parser.parse_args())