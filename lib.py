import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sklearn
import sklearn.decomposition


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


class ConvNet(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 n_classes=10,
                 ):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(1,32,kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
                )

        self.mlp1 = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(.5),
                nn.LazyLinear(n_classes),
                nn.Softmax()
                )

        self.layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.mlp1,
            ])

    def forward(self, x):
        for _,l in enumerate(self.layers):
            x = l(x)
        return x


def get_dataloaders(batch_size=4):
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader,testloader


def train_network(net,trainloader,testloader,epochs=10,print_acc=False): 
    # 4. Train the network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    net = net.to(DEVICE)
    net = net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times. Here 10 means 10 epochs
        running_loss = 0.0
        for i, (inputs,labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[epoch%d, itr%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if print_acc:
            test_accuracy = calc_accuracy(net,testloader)
            train_accuracy = calc_accuracy(net,trainloader)
            print('Epoch=%d Test Accuracy=%.3f' %
                   (epoch + 1, test_accuracy))
            print('Epoch=%d Train Accuracy=%.3f' %
                   (epoch + 1, train_accuracy))

    print('Finished Training')
    net = net.eval()

    test_accuracy = calc_accuracy(net,testloader)
    train_accuracy = calc_accuracy(net,trainloader)
    print('Test Accuracy=%.3f' %test_accuracy)
    print('Train Accuracy=%.3f' %test_accuracy)
    return net, train_accuracy, test_accuracy


def show_images(images):
    l = len(images)
    h = int(np.floor(np.sqrt(l)))
    w = int(np.ceil(l / h))
    
    _, ax = plt.subplots(h, w, figsize=(12, 12))
    for i in range(h):
        for j in range(w):
            idx = i*w + j
            if idx < l:
                ax[i, j].imshow(images[i*w + j],cmap='gray')
            ax[i, j].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def calc_accuracy(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu()==labels.cpu()).sum().item()

        return 100 * correct/total

class ModelVisualizer():
    def __init__(self, model, input_shape, device):
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def optimize_input(self, layer_no, target_index,epochs=200,start_img=None):
        if start_img == None:
            start_img = torch.rand(self.input_shape)

        start_img.requires_grad_(True)
        optimizer = torch.optim.Adam([start_img], lr=0.1, weight_decay=1e-6)

        for _ in range(1, epochs):
            optimizer.zero_grad()
            x = start_img.to(self.device)
            for index, layer in enumerate(self.model.layers):
                x = layer(x)
                if index == layer_no:
                    break
            out = x[0,target_index]
            loss = -torch.mean(out)
            loss.backward()
            optimizer.step()
        return start_img
