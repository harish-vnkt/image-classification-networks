import os
import argparse
import datetime

from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from LeNet import LeNet
from ResNet import Resnet


def run_lenet(args):

    train_set = datasets.CIFAR100("./data/", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  # pre-computed
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        )
    ]))
    test_set = datasets.CIFAR100("./data/", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  # pre-computed
            (0.5088, 0.4874, 0.4419),
            (0.2683, 0.2574, 0.2771)
        )
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.b, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    checkpoints_dir = "checkpoints/lenet/"
    final_dir = 'models/'

    for epoch in range(1, args.e + 1):
        loss_train = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

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

        if epoch % 10 == 0 or epoch == 0:

            checkpoint_path = os.path.join(checkpoints_dir, 'epoch_' + str(epoch) + '.pt')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path)

    model_path = os.path.join(final_dir, 'lenet.pth')
    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():

        correct = 0
        total = 0
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        print("Accuracy = {}".format(100 * (correct/total)))


def run_resnet(args):

    # The Resnet paper states the following transforms are applied on the train set
    train_set = datasets.CIFAR100("./data/", train=True, download=True, transform=transforms.Compose([
        transforms.Normalize(  # pre-computed
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        ),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ]))
    test_set = datasets.CIFAR100("./data/", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(  # pre-computed
            (0.5088, 0.4874, 0.4419),
            (0.2683, 0.2574, 0.2771)
        )
    ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.b, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.b, shuffle=True)

    checkpoints_dir = "checkpoints/resnet/"
    final_dir = 'models/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet(3).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    # learning rate should be decayed at 32k and 64k milestones
    scheduler = MultiStepLR(optimizer, milestones=[320, 480], gamma=0.1)

    for epoch in range(0, 640):
        loss_train = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            batch_loss = loss(predictions, labels)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss_train += batch_loss.item()

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(),
            epoch + 1,
            loss_train / len(train_loader)
        ))
        scheduler.step()

        if epoch % 100 == 0 or epoch == 0:
            checkpoint_path = os.path.join(checkpoints_dir, 'epoch_' + str(epoch) + '.pt')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()},
                       checkpoint_path)

    model_path = os.path.join(final_dir, 'lenet.pth')
    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():

        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

        print("Accuracy = {}".format(100 * (correct / total)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, dest='net', required=True, help='Type of network', choices=['lenet', 'resnet'])
    parser.add_argument('--batch_size', type=int, dest='b', required=False, default=128, help='Batch size for data loader')
    parser.add_argument('--learning_rate', type=float, dest='lr', required=False, default=1e-2, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=float, dest='e', required=False, default=100, help='Number of epochs for the training loop')
    args = parser.parse_args()

    if args.net == 'lenet':
        run_lenet(args)
    else:
        run_resnet(args)
