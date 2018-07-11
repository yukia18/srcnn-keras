from model import SRCNN
from utils import load_train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=33)
parser.add_argument('--c_dim', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--stride', type=int, default=None)
parser.add_argument('--scale', type=int, default=3)

def main(args):
    srcnn = SRCNN(
        image_size=args.image_size,
        c_dim=args.c_dim,
        is_training=True,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs)
    if args.stride is None:
        args.stride = args.image_size
    else:
        args.stride = args.stride
    X_train, Y_train = load_train(image_size=args.image_size, stride=args.stride, scale=args.scale)
    srcnn.train(X_train, Y_train)

if __name__ == '__main__':
    main(args=parser.parse_args())
