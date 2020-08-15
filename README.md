# Image Classification Networks on CIFAR-100

This repository contains code to train popular CNN architectures on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CNN architectures implemented are:

1. [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
2. [Resnet](https://arxiv.org/pdf/1512.03385.pdf)
3. [VGG-16](https://arxiv.org/pdf/1409.1556.pdf)

Note that VGG-16 does not have a training loop implemented because the authors trained it on the ImageNet dataset. However, the architecture is constructed keeping in mind the image size in CIFAR-100. This is a **personal project** for practicing Pytorch.

### Requirements

1. Python 3
2. Pytorch >= 1.1.0

### Running the code

The training loops for the LeNet and Resnet can be run through ```main.py```. The arguments to run the script can be viewed by typing ```python main.py -h```. The dataset is downloaded into a folder called ```data/``` using [torchvision](https://github.com/pytorch/vision). There is no need to predownload the dataset. The checkpoints are stored at regular intervals in the folder ```checkpoints/{model_name}```. The final model's state dictionary is stored in the ```models/``` folder.

In the Resnet paper, the authors have experimented with the architecture on CIFAR-10. The training in this implementation uses the same hyperparameters as in the experiment. If training arguments are provided for Resnet, they are ignored in the training loop. The hyperparameters include weight decay, momentum and adaptive training rate. The number of iterations mentioned in the paper are 64k, but we reduce that by a factor of 100. Additionally, the authors also suggest random flipping and a horizontal crop of 32 after 4 pixel padding, which are applied as transforms to the training set.
