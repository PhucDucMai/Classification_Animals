import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn as nn
from torch.optim import SGD, Adam, Adagrad
from CNN_Model import CNN
from dataset_animals import Dataset_Animal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP.*")


def get_args():
    parser = argparse.ArgumentParser(description='Animal Classifier')
    parser.add_argument('-p', '--data_path', type=str, default='./animals')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-s', '--image_size', type=int, default=224)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-r', '--trained_path', type=str, default="trained_models")
    args = parser.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="OrRd")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])

    # Train set , Valid set
    train_set = Dataset_Animal(root='./animals', train=True, transform=transform)
    valid_set = Dataset_Animal(root='./animals', train=False, transform=transform)

    # Load Model
    model = CNN(num_class=len(train_set.categories_animal))
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1

    batch_size = args.batch_size

    # DataLoader
    train_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
        'num_workers': 6
    }

    valid_params = {
        'batch_size': batch_size,
        'shuffle': False,
        'drop_last': False,
        'num_workers': 6
    }

    train_daloader = DataLoader(train_set, **train_params)
    valid_dataloader = DataLoader(valid_set, **valid_params)

    # Criterion and Optimizer

    folder_tensorboard = "tensorboard"

    if os.path.isdir(folder_tensorboard):
        shutil.rmtree(folder_tensorboard)
    os.mkdir(folder_tensorboard)
    writer = SummaryWriter(folder_tensorboard)

    # checkpoint_path = "checkpoint_path"
    if not os.path.isdir(args.trained_path):
        os.mkdir(args.trained_path)

    num_epoch = args.epoch

    # Train
    for epoch in range(start_epoch, num_epoch):
        losses = []
        model.train()
        progress_bar = tqdm(train_daloader, colour="blue")
        for iter, (image, label) in enumerate(progress_bar):
            # forward
            predictions = model(image)
            # across the loss
            loss_value = criterion(predictions, label)
            # backward and optimizer
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            losses.append(loss_value.item())

            progress_bar.set_description()

            progress_bar.set_description("Epoch: {}/{}. Loss value: {:.4f}".format(epoch + 1, len(train_daloader),
                                                                                   loss_value.item()))

            # write into file tensorboard: loss, iter, epoch, accuracy
            writer.add_scalar("Train/Loss", np.mean(losses), epoch * len(train_daloader) + iter)

        # Valid
        model.eval()
        losses_val = []

        all_predict = []
        all_label = []

        with (torch.no_grad()):
            for iter, (image, label) in enumerate(valid_dataloader):
                # forward
                predictions = model(image)
                # across the loss
                loss_value = criterion(predictions, label)

                losses_val.append(loss_value.item())

                # find index of max value in tensor predictions
                idex_max = torch.argmax(predictions, dim=1)

                all_predict.extend(idex_max)
                all_label.extend(label)

        writer.add_scalar("Val/Loss", np.mean(losses_val), epoch)
        acc = accuracy_score(all_label, all_predict)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        conf_matrix = confusion_matrix(all_label, all_predict)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.categories_animal))], epoch)

        check_point = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size
        }

        # save model
        torch.save(check_point, os.path.join(args.trained_path, "last.pt"))
        if acc > best_acc:
            torch.save(check_point, os.path.join(args.trained_path, "best.pt"))
            best_acc = acc

        scheduler.step()

    writer.close()


if __name__ == '__main__':
    args = get_args()
    train(args)
