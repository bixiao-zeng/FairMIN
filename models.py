#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn,flatten
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x)) # 10,320
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) # 10,50
        # return F.log_softmax(x, dim=1) #10,10
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=1)
        return x



class AlexNet(nn.Module):

    def __init__(self, args) -> None:
        super(AlexNet, self).__init__()
        self.Conv1 = nn.Conv2d(args.num_channels, 64, kernel_size=11, stride=4, padding=2)
        self.features = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x,dim=1)
        return x

class CNNusc_had(nn.Module):
    def __init__(self, args):
        super(CNNusc_had, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=(10,1))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(8,1))
        self.BatchNorm = nn.BatchNorm2d(10,momentum=0.1,affine=True,track_running_stats=True)
        self.BatchNorm2 = nn.BatchNorm2d(20, momentum=0.1, affine=True, track_running_stats=True)
        self.BatchNorm3 = nn.BatchNorm1d(50, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2040, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (5,1))
        x = F.relu(self.BatchNorm(x))   # torch.Size([10, 10, 18, 6])
        x = F.max_pool2d(self.conv2_drop(self.conv2(x)), (3,1))
        x = F.relu(x)   # torch.Size([10, 20, 3, 6])
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])   # torch.Size([10, 360])
        x = self.fc1(x) # torch.Size([10, 50])
        x = F.relu(x)   # torch.Size([10, 50])
        x = F.dropout(x, training=self.training)   # torch.Size([10, 50])
        x = self.fc2(x)   # torch.Size([10, 12])
        x = self.fc3(x)
        y = F.log_softmax(x,dim=1)
        return F.log_softmax(x, dim=1)

class CNNclothing(nn.Module):
    def __init__(self, args):
        super(CNNclothing, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3,3))
        self.BatchNorm = nn.BatchNorm2d(10,momentum=0.1,affine=True,track_running_stats=True)
        self.BatchNorm2 = nn.BatchNorm2d(20, momentum=0.1, affine=True, track_running_stats=True)
        self.BatchNorm3 = nn.BatchNorm1d(600,momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3920, 600)
        # self.fc2 = nn.Linear(2140,600)
        self.fc3 = nn.Linear(600, 50)
        self.fc4 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (5,5))
        x = F.relu(self.BatchNorm(x))   # torch.Size([10, 10, 18, 6])
        x = F.max_pool2d(self.conv2_drop(self.conv2(x)), (3,3))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])   # torch.Size([10, 5120])
        x = self.fc1(x) # torch.Size([10, 50])
        x = F.relu(self.BatchNorm3(x))   # torch.Size([10, 50])
        x = F.dropout(x, training=self.training)   # torch.Size([10, 50])
        x = self.fc3(x)
        x = F.dropout(x,training=self.training)
        x = self.fc4(x)
        x = F.dropout(x,training=self.training)
        y = F.log_softmax(x,dim=1)
        return x



class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
