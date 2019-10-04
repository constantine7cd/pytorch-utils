import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision.utils import make_grid


def show_dataset(dataset, n=6):
    imgs = torch.stack([dataset[i][0] for _ in range(n)
                        for i in range(3)])
    grid = make_grid(imgs, nrow=n).numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')


def plot_(x, y, title, label, marker='.'):
    plt.title(title)
    plt.xlabel(label)

    plt.plot(x, y, marker, label=label)

    plt.legend(loc='lower center')


def plot_training_history(training_history):
    eer_train = training_history['eer_train']
    eer_val = training_history['eer_val']
    losses = training_history['loss']

    eer_x = np.arange(len(eer_train)) + 1
    losses_x = np.arange(len(losses)) + 1

    plt.subplot(3, 1, 1)
    plot_(losses_x, losses, 'Training loss', 'Iteration', marker='o')

    plt.subplot(3, 1, 2)
    plot_(eer_x, eer_train, 'Training Equal Error Rate', 'Epoch', marker='-o')

    plt.subplot(3, 1, 3)
    plot_(eer_x, eer_val, 'Validation Equal Error Rate', 'Epoch', marker='-o')

    plt.gcf().set_size_inches(15, 15)
    plt.show()


def plot_v2(x, y, title, xlabel, label, marker='.'):
    plt.title(title)
    plt.xlabel(xlabel)

    plt.plot(x, y, marker, label=label)
    plt.grid()

    plt.legend()


def plot_training_history_v2(training_history):
    eer_train = training_history['eer_train']
    eer_val = training_history['eer_val']
    loss_train = training_history['loss_train']
    loss_val = training_history['loss_val']

    eer_x = np.arange(len(eer_train)) + 1
    losses_x = np.arange(len(loss_train)) + 1

    plt.subplot(2, 1, 1)
    plot_v2(losses_x, loss_train, 'Loss', 'Iteration', 'Train loss', marker='-o')
    plot_v2(losses_x, loss_val, 'Loss', 'Iteration', 'Validation loss', marker='-o')

    plt.subplot(2, 1, 2)
    plot_v2(eer_x, eer_train, 'Equal Error Rate', 'Epoch', 'EER train', marker='-o')
    plot_v2(eer_x, eer_val, 'Equal Error Rate', 'Epoch', 'EER val', marker='-o')

    plt.gcf().set_size_inches(15, 15)
    plt.show()


def plot_training_history_v3(s_dict, metric='eer_val'):
    if metric not in ['loss_val', 'loss_train', 'eer_val', 'eer_train']:
        raise ValueError('Incorrect value for metric')

    for i, (lr, dr) in enumerate(s_dict.keys()):
        eer_val = s_dict[(lr, dr)][metric]

        idx = i // 6 + 1

        plt.subplot(3, 1, idx)
        plot_v2(np.arange(len(eer_val)), eer_val, xlabel='epoch', title=metric,
                label=str(lr) + "_" + str(dr)[:4], marker='-')

    plt.gcf().set_size_inches(15, 25)

    plt.show()
