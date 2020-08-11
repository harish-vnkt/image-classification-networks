import os
import argparse
import datetime

from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn as nn

from LeNet import LeNet


def run(args):

    train_set = datasets.CIFAR100("./data/", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  # pre-computed
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        )
    ]))
    val_set = datasets.CIFAR100("./data/", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  # pre-computed
            (0.5088, 0.4874, 0.4419),
            (0.2683, 0.2574, 0.2771)
        )
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.b, shuffle=True)

    # networks = {'lenet': LeNet()}

    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(1, args.e + 1):
        loss_train = 0.0

        for images, labels in train_loader:

            predictions = model(images)
            batch_loss = loss(predictions, labels)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss_train += batch_loss.item()

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(),
            epoch,
            loss_train / len(train_loader)
        ))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--network', type=str, dest='net', required=True, help='Type of network', choices=['lenet', 'vggnet', 'resnet', 'inception'])
    parser.add_argument('--batch_size', type=int, dest='b', required=False, default=128, help='Batch size for data loader')
    parser.add_argument('--learning_rate', type=float, dest='lr', required=False, default=1e-2, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=float, dest='e', required=False, default=100, help='Number of epochs for the training loop')
    args = parser.parse_args()

    run(args)
