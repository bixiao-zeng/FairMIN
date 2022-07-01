from PIL import Image
from PIL import ImageFile
import numpy as np
from sklearn.model_selection import train_test_split
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
from collections import Counter
import scipy.io as sio
import gc

import torch
import sys
import time
import joblib

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class data_loader(Dataset):
    def __init__(self, samples, labels,args=None,mode='all',prob=[],ini_load=False):

        self.samples = samples
        self.labels = labels
        self.args = args
        self.mode = mode
        self.probability= prob
        if args.dataset == 'cifar10':
            self.samples = samples
            self.labels = labels
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_ref = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        camPath = 'data/camelyon17_v1.0'
        ImageFile.MAXBLOCK = 65536

        if self.mode=='labeled':
            img, target, prob = self.samples[index], self.labels[index], self.probability[index]
            if self.args.dataset == 'cifar10':
                img = Image.fromarray(img)
            else:
                img = Image.open(img).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img, target, prob = self.samples[index], self.labels[index], self.probability[index]
            if self.args.dataset == 'cifar10':
                img = Image.fromarray(img)
            else:
                img = Image.open(img).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2,target, prob
        elif self.mode=='all':
            img, target = self.samples[index], self.labels[index]
            if self.args.dataset == 'cifar10':
                img = Image.fromarray(img)
            else:
                img = Image.open(img).convert('RGB')
            img = self.transform(img)
            # if index>63 and index<73:
            #     im_data = img.numpy()
            #     im_data = im_data.transpose((1, 2, 0))
            #     plt.imshow(im_data)
            #     plt.savefig(os.path.join(str(index)+'_bat1.png'))
            #     plt.show()
            return img, target
        elif self.mode=='ref':
            img, target = self.samples[index], self.labels[index]
            if self.args.dataset == 'cifar10':
                img = Image.fromarray(img)
            else:
                img = Image.open(img).convert('RGB')
            img = self.transform_ref(img)
            # if index>63 and index<73:
            #     im_data = img.numpy()
            #     im_data = im_data.transpose((1, 2, 0))
            #     plt.imshow(im_data)
            #     plt.savefig(os.path.join(str(index)+'_bat1.png'))
            #     plt.show()
            return img, target

def cifar_assign(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = np.array(list(dict_users[i]))
    return dict_users


def bench_assign(dataset):

    len_dataset = len(dataset)
    num_items = int(len_dataset / 20)
    all_idxs =[i for i in range(len_dataset)]
    np.random.seed(42)

    sub_set = np.random.choice(all_idxs, num_items,
                                         replace=False)
    return sub_set

def get_dataset(args):

    train_data = []
    train_label = []
    if args.dataset == 'cifar10':

        root_dir = 'data/cifar-10/cifar-10-batches-py'
        for n in range(1, 6):
            dpath = '%s/data_batch_%d' % (root_dir, n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            train_label = train_label + data_dic['labels']
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))
        train_dataset = data_loader(train_data, np.array(train_label),args)

        test_dic = unpickle('%s/test_batch' % root_dir)
        test_data = test_dic['data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic['labels']
        test_dataset = data_loader(test_data, np.array(test_label),args,mode='ref')



    return train_dataset, test_dataset

if __name__=='__main__':
    train_data = []
    train_label = []
    root_dir = 'data/cifar-10/cifar-10-batches-py'
    for n in range(1, 6):
        dpath = '%s/data_batch_%d' % (root_dir, n)
        data_dic = unpickle(dpath)
        train_data.append(data_dic['data'])
        train_label = train_label + data_dic['labels']
    train_data = np.concatenate(train_data)
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((2, 3, 1,0))
    noise_data = sio.loadmat('data/SVHN/train_32x32.mat')
    noise_data = noise_data['X']
    for i in range(len(train_data)):
        if train_label[i]==0 or train_label[i]==1 or train_label[i]==9:
            plt.imshow(train_data[:, :, :, i])
            plt.savefig(os.path.join('plots', 'label_'+str(train_label[i])+'_'+str(i)+'.png'))
            plt.show()

