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
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
import graphviz
import copy


def get_feature_names():
    """
    Feature names for lines, circles, corners
    with max_features = 6
    """
    names = []
    for i in range(1,7):
        for f in ['x1','y1','x2','y2']:
            names.append(f"L{i} {f}")
    for i in range(1,7):
        for f in ['x','y','r']:
            names.append(f"Ci{i} {f}")
    for i in range(1,7):
        for f in ['x','y']:
            names.append(f"Co{i} {f}")
    return names

def bw_to_rgb(img):
    """
    Expand bw mnist img to rgb
    """
    i = np.zeros((28,28,3))
    i[:,:,0] = img
    i[:,:,1] = img
    i[:,:,2] = img
    return np.uint8(i)

def draw_lines(img,lines,color=(255,0,0)):
    i = copy.deepcopy(img)
    for l in lines:
        if np.isnan(l).any():
            break
        l = np.uint8(l)
        x1,y1,x2,y2 = l[0],l[1],l[2],l[3]
        cv.line(i, (x1,y1),(x2,y2), color, thickness=1, lineType=8) 
    return i

def draw_circles(img,circles,color=(255,0,0)):
    i = copy.deepcopy(img)
    for c in circles:
        if np.isnan(c).any():
            break
        c = np.uint8(c)
        x,y,r = c[0],c[1],c[2]
        cv.circle(i, (x,y),r, color, thickness=1, lineType=8) 
    return i

def draw_corners(img,corners,color=(255,0,0)):
    i = copy.deepcopy(img)
    for c in corners:
        if np.isnan(c).any():
            break
        c = np.uint8(c)
        x,y, = c[0],c[1]
        cv.line(i, (x,y),(x,y), color, thickness=1, lineType=8) 
    return i

def show_overlay(base_img,lines,circles,corners):
    """
    Find a way to draw lines,circles,corners
    on top of an mnist image in different colours
    """
    img = bw_to_rgb(base_img)
    img = draw_lines(img,lines,color=(255,0,0))
    img = draw_circles(img,circles,color=(0,255,0))
    img = draw_corners(img,corners,color=(0,0,255))
    plt.title('MNIST Image overlay')
    plt.imshow(img)
    plt.show()

def get_lines(img,max_lines=6):
    """
    input: mnist image: (28,28)
    out: four lines in (x1,y1,x2,y2) form: (max_lines,4)
    if less than max_lines lines found,
    fill with np.nan 
    """
    lines = cv.HoughLinesP(img,rho = 1,theta = 1*np.pi/180,threshold = 20)
    res = np.full((max_lines,4),np.nan)
    if lines is not None:
        n_lines = min(max_lines,lines.shape[0])
        lines = np.array(lines).reshape(-1,4)
        res[:n_lines,:] = lines[:n_lines,:]
    return res

def get_circles(img,max_circles=6):
    """
    input: mnist image: (28,28)
    out: two circles in (x,y,r) (max_circles,3)
    if less than max_circles circles found,
    fill with np.nan
    """
    n,_ = img.shape
    circles = cv.HoughCircles(
            img,
            cv.HOUGH_GRADIENT,
            1,
            n/8,
            param1=50,
            param2=10,
            minRadius=1,
            maxRadius=20
            )
    res = np.full((max_circles,3),np.nan)
    if circles is not None:
        n_circles = min(max_circles,circles.shape[0])
        circles = np.array(circles).reshape(-1,3)
        res[:n_circles,:] = circles[:n_circles,:]
    return res

def get_corners(img,thresh=155,max_corners=6):
    """
    input: mnist image: (28,28)
    out: 5 circles in (x,y) form: (5,2)
    if less than two circles found,
    replace with np.nan for xgboost
    """
    # use cv.cornerHarris
    # sort by x,y
    corners = cv.cornerHarris(img,2,3,0.04)
    res = np.full((max_corners,2),np.nan)
    if corners is not None:
        n,x = np.min(corners),np.max(corners)
        corners = np.add(corners,-n) / np.add(x,-n) * 255
        corners = np.array((corners > thresh).nonzero()).T
        corners = np.flip(corners,axis=1)

        n_corners = min(max_corners,corners.shape[0])
        res[:n_corners,:] = corners[:n_corners,:]
    return res

def get_features(img,max_features=6):
    # Flatten everything out into a single feature vector
    lines = get_lines(img,max_features).flatten()
    circles = get_circles(img,max_features).flatten()
    corners = get_corners(img,max_features).flatten()
    return np.concatenate((lines,circles,corners))

def show_mnist(img,title='MNIST Image'):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch device:", DEVICE)

class ConvNet(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 n_classes=10,
                 ):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(1,32,kernel_size),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size),
                nn.LeakyReLU(),
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

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
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
    print('Train Accuracy=%.3f' %train_accuracy)
    return net, train_accuracy, test_accuracy


def show_images(images,cmap='gray'):
    l = len(images)
    h = int(np.floor(np.sqrt(l)))
    w = int(np.ceil(l / h))

    _, ax = plt.subplots(h, w, figsize=(12, 12))
    for i in range(h):
        for j in range(w):
            idx = i*w + j
            if idx < l:
                ax[i, j].imshow(images[i*w + j],cmap=cmap)
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
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = device

    def optimize_input(self, layer_no, target_index,epochs=200,lr=0.1,start_img=None):
        if start_img == None:
            start_img = torch.rand(self.input_shape)

        start_img.requires_grad_(True)
        optimizer = torch.optim.Adam([start_img], lr=lr, weight_decay=1e-6)

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

    def get_saliency(self, img, label=None):
        img = img.to(DEVICE).reshape(self.input_shape)
        img.requires_grad_(True)
        out = self.model(img)
        # Get saliency for predicted class
        if label == None:
            pr,_ = torch.max(out,dim=1)
            pr.backward()
        # Saliency for specified class
        else:
            pr = out[:,label]
            pr.backward()
        slc = img.grad.cpu()
        slc = (slc - slc.min())/(slc.max()-slc.min())
        return slc.reshape(28,28)

def visualize_saliency(viz,img,label):
    slc = viz.get_saliency(img)
    _, ax = plt.subplots(1, 2, figsize=(24, 24))
    ax[0].imshow(img.reshape(28,28),cmap='gray')
    ax[0].text(0.,-1.,f'Input ({label})',size=20)
    ax[0].axis('off')
    ax[1].imshow(slc,cmap='coolwarm')
    ax[1].text(0.,-1.,'Saliency',size=20)
    ax[1].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
