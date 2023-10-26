import argparse
import os
import cv2
import torch
from CNN_Model import CNN
import torch.nn as nn
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Animal Classifier')
    parser.add_argument('-p', '--data_path', type=str, default='./animals')
    parser.add_argument('-s', '--image_size', type=int, default=224)
    parser.add_argument('-c', '--checkpoint_path', type=str, default='./trained_models/best.pt')
    parser.add_argument('-i', '--image_path', type=str, default='./image_test/cat1.jpeg')
    arg = parser.parse_args()
    return arg


def test(args):
    categories_test = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    model = CNN(num_class=len(categories_test))
    if os.path.isfile(args.checkpoint_path) and args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("File does not exist")
        exit(0)

    if not args.image_path:
        print("image must be provided")
        exit(0)

    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(float)
    image /= 255
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()

    soft_max = nn.Softmax()

    with torch.no_grad():
        prediction = model(image)

    prob = soft_max(prediction)
    max_value, indexof_max = torch.max(prob, dim=1)
    print("This image about {} with probability of {:.4f}".format(categories_test[indexof_max], max_value.item()))


if __name__ == '__main__':
    args = get_args()
    test(args)
