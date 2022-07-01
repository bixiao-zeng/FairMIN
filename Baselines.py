#!/usr/bin/env Python
# coding=utf-8
# encoding:UTF-8

from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
from torch.utils.tensorboard import SummaryWriter
import pdb
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as tfs
from edl_losses import *
import warnings
from pathlib import Path
from sklearn.preprocessing import normalize

import copy
from models import *

from torch.optim.lr_scheduler import StepLR
from scipy import stats, misc
from operator import itemgetter
from kneed import KneeLocator
from sklearn.ensemble import IsolationForest
import pickle
import joblib
import csv
# from feature_selector import FeatureSelector
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from dataprocessing import *
from numpy import *
import copy
import scipy.io as sio
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from loguru import logger
from itertools import permutations

# import torch_mlu.core.mlu_model as ct

cuda_device = "0"
os.environ['CUDA_VISIBLE_DEVICES']=cuda_device

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--method', default='sfavg_all', help='baselines')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--on', default=0.5, type=float, help='open noise ratio')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noisy_dataset', default='SVHN', type=str)
parser.add_argument('--autoload', default=False, type=bool)
parser.add_argument('--load_loss', default=True, type=bool)
parser.add_argument('--mode', default='sym', type=str)
parser.add_argument('--machine',default=0,type=int,help='0 represents nvidia and 1 represents hwj')
parser.add_argument('--noisy_clnt', default=0.25,type=float, help='the ratio of noisy client')
parser.add_argument('--prepare_data', default=False, type=bool)
#=========================not so common==================================================
parser.add_argument('--local_bs', default=32, type=int, help='train batchsize')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--start_epoch', default=1, type=int)  #
parser.add_argument('--num_users', default=4, type=int)
parser.add_argument('--train_epochs_C', default=10)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--mom', type=int, default=0.9,
                    help="momentem for optimizer")  # 0.9
parser.add_argument('--decay', type=int, default=5e-4,
                    help="momentem for optimizer")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--drp', default=False, type=bool)
parser.add_argument('--alpha', default=0.75, type=float, help='hyperparameter for SFedAvg')
parser.add_argument('--beta', default=0.25, type=float, help='hyperparameter for SFedAvg')
parser.add_argument('--m', default=3, type=float, help='hyperparameter for SFedAvg')
parser.add_argument('--R', default=6, type=float, help='hyperparameter for SFedAvg')

args = parser.parse_args()

def args_groupset():
    global args
    if args.dataset == 'camelyon17':
        args.noisy_dataset = 'Monusac'
        args.start_epoch = 1
        args.local_bs = 32
        args.num_users = 5
        args.num_classes = 2
        args.slr = 0.001
        args.dlr = 0.001
        args.drp = True
        args.rev_score = False

args_groupset()
set_noyclnt = 0.25
num_noiy = int(set_noyclnt*args.num_users) #脏客户端的个数
pt = int(args.num_users/num_noiy)
noise_idx = [True if i%pt==0 else False for i in range(1,args.num_users+1)]
see = np.sum(noise_idx)
noy_clnt = see/args.num_users
print('the number of clients: '+str(args.num_users))
print('the nuber of noisy clients: '+str(see))
right_closed,real_closed,pred_closed = 0,0,0
right_clean,real_clean,pred_clean = 0,0,0
right_noise, real_noise, pred_noise = 0,0,0
sv,rev,par,clnt_selec = [0]*args.num_users,[1/args.num_users]*args.num_users,[1/args.num_users]*args.num_users,[0]*args.num_users
FLI_Y,FLI_Q, FLI_C= [0]*args.num_users,[0]*args.num_users,[0]*args.num_users

writer = SummaryWriter(comment='scalar')

args.title = '{}-{}-{}'.format(args.dataset, args.noisy_dataset,args.mode)
config = 'noise={}_{}_numusr={}'.format(args.r,args.on,args.num_users)
data_dir = os.path.join('data/'+args.title,config)
checkpoint_dir = os.path.join('saveDicts/'+args.method+'/'+args.title,config)
plots_dir = os.path.join('plots/'+args.method+'/'+args.title, config)
# ------------------------------------------------
Path(os.path.join(checkpoint_dir, )).mkdir(parents=True, exist_ok=True)
Path(os.path.join(data_dir, )).mkdir(parents=True, exist_ok=True)
Path(os.path.join(plots_dir, )).mkdir(parents=True, exist_ok=True)
serverPath = os.path.join(data_dir, 'server')
clientPath = os.path.join(data_dir, 'client')
Path(serverPath).mkdir(parents=True, exist_ok=True)
Path(clientPath).mkdir(parents=True, exist_ok=True)

noise_chart_op = [[] for i in range(args.num_users)]
noise_chart_cl = [[] for i in range(args.num_users)]
global_weight_collector = []

logger.add(os.path.join(checkpoint_dir,'runtime.log'))


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


subjective_loss = edl_mse_loss
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()


class Server():
    def __init__(self, model_C, model_B, datapath):
        self.model_C = model_C
        self.model_B = model_B
        self.data_dir = datapath
        self.weights_C = []
        self.stop = False
        with open(os.path.join(self.data_dir, 'testdata.pkl'), 'rb') as f:
            test_dataset = joblib.load(f)
        self.test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False,num_workers=args.num_workers,drop_last=args.drp)
        with open(os.path.join(self.data_dir, 'benchdata.pkl'), 'rb') as f:
            bench_dataset = joblib.load(f)
        self.bench_loader = DataLoader(bench_dataset, batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp)
        global noise_chart_op,noise_chart_cl
        with open(os.path.join(serverPath, 'noise_chart_op.pkl'), 'rb') as f:
            noise_chart_op = pickle.load(f)
        with open(os.path.join(serverPath, 'noise_chart_clo.pkl'), 'rb') as f:
            noise_chart_cl = pickle.load(f)
        global allNoise_c,allNoise_o

        if args.method == 'DS':
            all_idxs = [i for i in range(len(bench_dataset))]
            idx_train = np.random.choice(all_idxs, int(len(bench_dataset) / 10) * 8,
                                         replace=False)
            idx_test = np.array(list(set(all_idxs) - set(idx_train)))

            x_bench = bench_dataset.samples
            y_bench = bench_dataset.labels
            train_bench = data_loader(x_bench[idx_train], y_bench[idx_train], args,mode='all')
            test_bench = data_loader(x_bench[idx_test], y_bench[idx_test], args,mode='ref')
            self.bench_trloader = DataLoader(train_bench, batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp)
            self.bench_tstloader = DataLoader(test_bench, batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp)
            savepath_dir = os.path.join('saveDicts/'+args.method+'/'+args.title,'benchmodel_frombentr_0.8')
            savepath_file = os.path.join('saveDicts/'+args.method+'/'+args.title, 'benchmodel_frombentr_0.8_197ep.json')
            if not os.path.exists(savepath_file):
                self.pre_train(savepath=savepath_dir)
            self.model_C.load_state_dict(torch.load(savepath_file,map_location=torch.device('cpu')))

    def test(self, epoch):
        model = self.model_C
        model.eval()
        correct = 0
        total = 0
        LOSS = 0
        criterion = nn.CrossEntropyLoss().to(args.device)
        num_iter = len(self.test_loader)

        with torch.no_grad():
            for batch_ixx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                LOSS += loss

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                sys.stdout.write('\r')
                sys.stdout.write('Test  | Global Epoch %d | Iter[%3d/%3d]\t loss: %.4f'
                                 % (epoch, batch_ixx + 1, num_iter,
                                    loss.item()))
                sys.stdout.flush()
                del inputs, targets, outputs, loss
                if args.machine:
                    ct.empty_cached_memory()
                else:
                    torch.cuda.empty_cache()
        acc = 100. * correct / total

        logger.debug("\n|Global Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
        writer.add_scalar('Test/Accuracy', acc, epoch)

        del acc, LOSS

    def val(self,model,pm,clnt):
        model.eval()
        correct = 0
        total = 0
        LOSS = 0
        criterion = nn.CrossEntropyLoss().to(args.device)
        num_iter = len(self.bench_loader)
        with torch.no_grad():
            for batch_ixx, (inputs, targets) in enumerate(self.bench_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                LOSS += loss
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                sys.stdout.write('\r')
                sys.stdout.write('Val |Permutation: %s |Client: %d |Iter[%3d/%3d]\t loss: %.4f'
                                 % (pm,clnt,batch_ixx, num_iter,
                                    loss.item()))
                sys.stdout.flush()
                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()
        acc =  correct / total
        return acc


    def pre_train(self, max_epochs=200, savepath=''):
        self.model_C.train()
        criterion = nn.CrossEntropyLoss().to(args.device)

        loader = self.bench_loader
        num_iter = (len(loader.dataset) // loader.batch_size) + 1
        optimizer = optim.SGD(self.model_C.parameters(), lr=0.01, momentum=args.mom, weight_decay=args.decay)
        best_acc = 0
        for epoch in range(max_epochs):
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                outputs = self.model_C(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write('Server  | Pre-train | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (epoch, max_epochs, batch_idx + 1, num_iter,
                                    loss.item()))
                sys.stdout.flush()
            # scheduler.step()
            accD, lossD = self.test(epoch  - max_epochs, 'netD')
            writer.add_scalar('Test/Accuracy', accD, epoch - max_epochs)
            writer.add_scalar('Test/Loss', lossD, epoch - max_epochs)
            if accD>best_acc:
                torch.save(self.model_C.state_dict(), savepath + '_' + str(epoch) + 'ep.json')
                best_acc = accD

    def FedAvg(self, w, userdatlen,cover=True):
        if len(w) == 107:
            return w
        else:
            if args.method == 'FedNS':
                self.weights_C = copy.deepcopy(w)
            w_avg = copy.deepcopy(w[0])
            sum = np.sum(userdatlen)
            for key in w_avg.keys():
                for i in range(len(w)):
                    a = w[i][key]
                    b = userdatlen[i] / sum
                    tensor = torch.mul(w[i][key], userdatlen[i] / sum)
                    if i == 0:
                        w_avg[key] = tensor.type(w[i][key].dtype)
                    else:
                        w_avg[key] += tensor.type(w[i][key].dtype)
            if cover:
                self.model_C.load_state_dict(w_avg)
            else:
                return w_avg



    def Sfavg(self,w,userdatlen,R_mont=args.R):
        w = np.array(w)
        userdatlen = np.array(userdatlen)
        a_diff = [0]*args.num_users
        global sv,rev,par
        k = 0
        for perm in permutations(clnt_selec):
            for i in clnt_selec:
                idx = list(perm).index(i)
                pre_i = perm[:idx+1]
                pre_i = [True if j in pre_i else False for j in range(args.num_users)]
                w_withi = self.FedAvg(w[pre_i],userdatlen=userdatlen[pre_i],cover=False)
                self.model_B.load_state_dict(w_withi)
                a_withi = self.val(self.model_B,pm=str(k)+'_0',clnt=i)
                if idx == 0:
                    a_noi = 0
                else:
                    no_i = perm[:idx]
                    no_i = [True if j in no_i else False for j in range(args.num_users)]
                    w_noi = self.FedAvg(w[no_i],userdatlen=userdatlen[no_i],cover=False)
                    self.model_B.load_state_dict(w_noi)
                    a_noi = self.val(self.model_B,pm=str(k)+'_1',clnt=i)
                a_diff[i] += a_withi-a_noi
                if k == R_mont-1:
                    sv[i] += a_diff[i]/R_mont
                    rev[i] = args.alpha*rev[i]+args.beta*sv[i]
            k+=1
            if k == R_mont:
                break
        selec = [True if j in clnt_selec else False for j in range(args.num_users)]
        par = np.exp(rev) / sum(np.exp(rev))
        if args.method == 'soft_sfavg':
            userdatlen = [userdatlen[i] * par[i] for i in range(args.num_users)]
            userdatlen = np.array(userdatlen)
        with open(os.path.join(serverPath, 'shapley_value.pkl'), 'wb') as f:
            pickle.dump(sv, f)
        self.FedAvg(w[selec],userdatlen=userdatlen[selec])

    def LabUnion(self,w,userdatlen):
        w = np.array(w)
        userdatlen = np.array(userdatlen)
        a_diff = [0] * args.num_users
        global sv, rev,par
        for i in clnt_selec:
            idx = clnt_selec.index(i)
            pre_i = clnt_selec[:idx + 1]
            pre_i = [True if j in pre_i else False for j in range(args.num_users)]
            w_withi = self.FedAvg(w[pre_i], userdatlen=userdatlen[pre_i], cover=False)
            self.model_B.load_state_dict(w_withi)
            a_withi = self.val(self.model_B, pm= 'coalition_0', clnt=i)
            if idx == 0:
                a_noi = 0
            else:
                no_i = clnt_selec[:idx]
                no_i = [True if j in no_i else False for j in range(args.num_users)]
                w_noi = self.FedAvg(w[no_i], userdatlen=userdatlen[no_i], cover=False)
                self.model_B.load_state_dict(w_noi)
                a_noi = self.val(self.model_B, pm='coalition_1', clnt=i)
            a_diff[i] = a_withi - a_noi
            rev[i] = a_diff[i]
        par = np.exp(rev) / sum(np.exp(rev))
        selec = [True if j in clnt_selec else False for j in range(args.num_users)]
        self.FedAvg(w[selec], userdatlen=userdatlen[selec])

    def Individual(self,w,userdatlen):
        w = np.array(w)
        userdatlen = np.array(userdatlen)
        global sv, rev,par
        for i in clnt_selec:
            w_i = self.FedAvg(w[i], userdatlen=userdatlen[i], cover=False)
            self.model_B.load_state_dict(w_i)
            rev[i] = self.val(self.model_B, pm= 'individual', clnt=i)
        par = np.exp(rev) / sum(np.exp(rev))
        selec = [True if j in clnt_selec else False for j in range(args.num_users)]
        self.FedAvg(w[selec], userdatlen=userdatlen[selec])

    def FLI(self,w,userdatlen,omega=10,increment=0.5):
        w = np.array(w)
        userdatlen = np.array(userdatlen)
        global rev,par,FLI_C, FLI_Q, FLI_Y
        lamda = [0]*args.num_users
        for i in clnt_selec:
            w_i = self.FedAvg(w[i], userdatlen=userdatlen[i], cover=False)
            self.model_B.load_state_dict(w_i)
            rev[i] = self.val(self.model_B, pm='FLI', clnt=i)
            FLI_C[i] = rev[i]/2
            if FLI_Y[i]>0:
                lamda[i] = increment
            else:
                lamda[i] = 0
            rev[i] = 0.5*(omega*rev[i]+FLI_Y[i]+FLI_C[i]+FLI_Q[i]+lamda[i])
        sum_u = sum(rev)
        par = [rev[i]/sum_u for i in range(args.num_users)]
        for i in clnt_selec:
            FLI_Y[i] = max(0,FLI_Y[i]+FLI_C[i]-par[i])
            FLI_Q[i] = max(0,FLI_Q[i]+lamda[i]-par[i])
        data_save = [rev,FLI_C,FLI_Y,FLI_Q]
        with open(os.path.join(serverPath, 'media-result.pkl'), 'wb') as f:
            pickle.dump(data_save, f)
        selec = [True if j in clnt_selec else False for j in range(args.num_users)]
        self.FedAvg(w[selec], userdatlen=userdatlen[selec])

    def cmp_bench_loss(self):
        device = args.device
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        bench_loss = []
        for idx, (data, label) in enumerate(self.bench_tstloader):
            data, label = data.to(device), label.to(device)
            outputs = self.model_C(data)
            loss = criterion(outputs, label)
            bench_loss.extend(loss.tolist())
            del data,label,outputs,loss
            if args.machine:
                ct.empty_cached_memory()
            else:
                torch.cuda.empty_cache()
        # 
        bench_loss = np.array(bench_loss)
        bench_loss = (bench_loss-bench_loss.min())/(bench_loss.max()-bench_loss.min())
        return bench_loss

    def sendmodel(self):
        torch.save({
            'netC_state_dict': self.model_C.state_dict(),
        }, os.path.join(checkpoint_dir, 'servermodel_ep.json'))

    def compute_lamda(self, samp1=[], samp2=[], log=True):


        user_len = len(samp2)
        predClean, predOpen, predClosed = [False] * user_len, [False] * user_len, [False] * user_len
        enu1 = sorted(enumerate(samp1), key=itemgetter(1))
        enu2 = sorted(enumerate(samp2), key=itemgetter(1))

        sp1 = [value for index, value in enu1]
        sp2 = [value for index, value in enu2]
        temp = -1

        D = stats.ks_2samp

        inixis = D(samp1, samp2)
        inixis = inixis[0]

        # 设定步长加速KS距离计算
        distance = []
        stp = 100
        STEP = np.arange(user_len / stp, user_len + 1, user_len / stp)
        STEP = STEP.astype(np.int)
        idx = 0
        for t in STEP:
            trunsp2 = sp2[:t]

            dis = D(sp1, trunsp2)
            dis = dis[0]
            distance.append(dis)
            if dis <= inixis:
                inixis = dis
                temp = t - 1  
                best_idx = idx
            if log:
                writer.add_scalar('KS_loss', dis, t)
            idx += 1

        kneedle_cov_dec = KneeLocator(STEP[:best_idx], distance[:best_idx], curve='convex', direction='decreasing',
                                      online=True)
        kneedle_cov_dec2 = KneeLocator(STEP, distance, curve='convex', direction='decreasing',
                                      online=True)
        kneedle_cov_dec2.plot_knee()
        plt.savefig('Knee_KS_0929.png')

        diff_max = np.max(kneedle_cov_dec.y_difference)
        # 
        signific = 0.05
        # signific = 0
        if diff_max < signific:
            tag1 = temp
        else:
            tag1 = kneedle_cov_dec.knee
        tag2 = temp
        Index2 = [index for index, value in enu2]
        for i in range(user_len):
            if i <= tag1:
                predClean[Index2[i]] = True
            elif tag1 < i <= tag2:
                predOpen[Index2[i]] = True
            else:
                predClosed[Index2[i]] = True

        return [np.array(predClean), np.array(predOpen), np.array(predClosed)], tag1, tag2, sp2[tag1].item(), sp2[
            tag2].item()


class Client:
    def __init__(self, client_id, model_C, model_S, datapath=[]):
        self.model_C = model_C
        self.data_dir = datapath
        self.client_id = client_id
        self.avai_dataset = None
        self.keys = []
        self.clean_labels = []

        with open(os.path.join(self.data_dir, 'dataset' + str(client_id) + '.pkl'), 'rb') as f:
            self.dataset = joblib.load(f)

        if noise_idx[self.client_id]:
            self.dataset.args = args
        ###
        self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=args.local_bs,num_workers=args.num_workers,drop_last=args.drp)
        with open(os.path.join(serverPath, 'testdata.pkl'), 'rb') as f:
            test_dataset = joblib.load(f)
        ###
        self.test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False,num_workers=args.num_workers,drop_last=args.drp)
        with open(os.path.join(clientPath, 'clean_labels' + str(client_id) + '.pkl'), 'rb') as f:
            self.clean_labels = joblib.load(f)


    def receivemodel(self):
        torchLoad = torch.load(os.path.join(checkpoint_dir, 'servermodel_ep.json'))
        self.model_C.load_state_dict(torchLoad['netC_state_dict'])



    def lamda1_detect(self, lamda):
        # file_path = os.path.join(clientPath,'DS_dataset%d_lamda%.2f.pkl'%(self.client_id,lamda))
        global right_clean,real_clean,pred_clean
        global right_closed,real_closed,pred_closed
        global right_noise,real_noise,pred_noise
        avai_idx = []
        nonavai_idx = []
        #
        for i, v in enumerate(self.local_loss):
            if v <= lamda:
                avai_idx.append(i)
            else:
                nonavai_idx.append(i)
        actual_num = len(self.local_loss)
        # performance of clean
        clean, opens, closed = self.get_noise()
        clean = np.array(clean)
        clean_pred = clean[avai_idx]
        right_clean += np.sum(clean_pred)
        real_clean += np.sum(clean[:actual_num])
        pred_clean += len(avai_idx)

        closed = np.array(closed)
        closed_pred = closed[nonavai_idx]
        right_closed += np.sum(closed_pred)
        real_closed += np.sum(closed[:actual_num])
        pred_closed += len(nonavai_idx)

        opens = np.array(opens)
        right_noise += np.sum(closed_pred)+np.sum(opens[nonavai_idx])
        real_noise += np.sum(closed[:actual_num])+np.sum(opens[:actual_num])
        pred_noise += len(nonavai_idx)

        if noise_idx[self.client_id]:
            recall_clean = round(right_clean / real_clean,4)
            precision_clean = round(right_clean / pred_clean,4)
            f1score_clean = round((recall_clean + precision_clean) / 2,4)

            recall_closed = round(right_closed/real_closed,4)
            precision_closed = right_closed/pred_closed
            f1score_closed = (recall_closed+precision_closed)/2

            recall_noise = round(right_noise/real_noise,4)
            precision_noise = round(right_noise/pred_noise,4)
            f1score_noise = round((recall_noise+precision_noise)/2,4)
            debug = True
        x = self.dataset.samples
        y = self.dataset.labels
        newx = x[avai_idx]
        newy = y[avai_idx]
        self.avai_dataset = data_loader(newx, newy, args)

    def reference(self):

        self.model_C.eval()
        device = args.device
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        B_predits = []
        Loss = []
        for batch_idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(device), label.to(device)

            outputs = self.model_C(data)

            loss = criterion(outputs, label)
            Loss.extend(loss.tolist())
            # _, predits = torch.max(outputs, 1)
            sys.stdout.write('\r')
            sys.stdout.write('batch %d / %d'%(batch_idx,len(self.data_loader)))
            sys.stdout.flush()
            # predits = predits.cpu().detach().numpy()
            # B_predits.extend(predits)
            del loss,outputs,data,label
            if args.machine:
                ct.empty_cached_memory()
            else:
                torch.cuda.empty_cache()
        Loss = np.array(Loss)
        # 
        Loss = (Loss-Loss.min())/(Loss.max()-Loss.min())
        self.local_loss = Loss
        return B_predits, Loss

    def model_train_val(self, epoch, train_set, val_set):
        model = copy.deepcopy(self.model_C)
        device = args.device
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        criterion_red = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.local_mom,
                                    weight_decay=args.local_decay)
        batVerbose = 9
        batch_nm = int(len(train_set.dataset) / args.local_bs)
        model.train()
        bestacc = 0
        acc_ls = []
        for iter in range(epoch):

            print('Epoch {}/{}'.format(iter + 1, epoch))
            print('-' * 10)
            batch_loss = []
            running_corrects = 0
            model.train()
            for batch_idx, (data, label) in enumerate(train_set):
                data, label = data.to(device), label.to(device)
                model.zero_grad()
                outputs = model(data)
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == label).item()
                lossback = criterion_red(outputs, label)

                lossback.backward()
                optimizer.step()
                if batch_idx % batVerbose == 0:
                    print('| Server Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data),
                        len(train_set.dataset), 100. * batch_idx / batch_nm, lossback.item()))

                batch_loss.append(lossback.item())
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(val_set):
                    data, label = data.to(device), label.to(device)
                    outputs = model(data)
                    _, preds = torch.max(outputs.data, 1)
                    cross_entropy = criterion_red(outputs, label)
                    correct += torch.sum(torch.eq(preds, label)).item()
            val_acc = correct / len(val_set.dataset)
            acc_ls.append(val_acc)
            if val_acc > bestacc:
                bestacc = val_acc
                bestmodel = copy.deepcopy(model)

        bestmodel.eval()
        PREDS = []
        CROSS_ENTROPY = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(val_set):
                data, label = data.to(device), label.to(device)
                log_probs = bestmodel(data)
                _, preds = torch.max(log_probs.data, 1)
                preds_ls = preds.tolist()
                PREDS.extend(preds_ls)
                cross_entropy = criterion(log_probs, label)
                CROSS_ENTROPY.extend(cross_entropy.tolist())

        return PREDS, CROSS_ENTROPY


    def test(self, epoch, mod_name=''):
        if mod_name == 'netS':
            model = self.model_S
        else:
            model = self.model_C
        model.eval()
        correct = 0
        total = 0
        LOSS = 0
        criterion = nn.CrossEntropyLoss().to(args.device)
        num_iter = len(self.test_loader)
        with torch.no_grad():
            for batch_ixx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                LOSS += loss

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                sys.stdout.write('\r')
                sys.stdout.write('TEST | User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (self.client_id, epoch, batch_ixx + 1, num_iter,
                                    loss.item()))
                sys.stdout.flush()
                del inputs, targets, outputs, loss
                if args.machine:
                    ct.empty_cached_memory()
                else:
                    torch.cuda.empty_cache()

        acc = 100. * correct / total
        LOSS = LOSS / len(self.test_loader)

        # print("\n| Global Epoch #%d|\t Client %s |\t Accuracy: %.2f%%\n" % (epoch,self.client_id, acc))
        logger.debug("\n| Global Epoch #%d|\t Client %s |\t Accuracy: %.2f%%\n" % (epoch,self.client_id, acc))

        return acc

    def get_noise(self):
        data_len = len(self.dataset)
        if noise_idx[self.client_id]:
            open = noise_chart_op[self.client_id]
            close = noise_chart_cl[self.client_id]
            clean = [False] * data_len
            for i in range(data_len):
                if not open[i] and not close[i]:
                    clean[i] = True
            clean = np.array(clean)
        else:
            open = [False] * data_len
            close = [False] * data_len
            clean = [True] * data_len
        return clean, open, close

    def one_hot_embedding(self, labels, num_classes=10):
        y = torch.eye(num_classes)
        neg = labels < 0  # negative labels
        labels[neg] = 0  # placeholder label to class-0
        y = y[labels]  # create one hot embedding
        y[neg, 0] = 0  # remove placeholder label
        return y

    def sce_loss(self, outputs, labels, reduce=True):
        sfm_probs = F.softmax(outputs)  # 添加激活层取正数
        loss_ce = F.nll_loss(F.log_softmax(outputs), labels, reduce=reduce)
        q = self.one_hot_embedding(labels,num_classes=args.num_classes).to(args.device)
        q = torch.clamp(q,min=1e-4,max=1)
        q = torch.log10(q)
        multi = q.mul(sfm_probs)
        # np.multiply(q,sfm_probs.cpu().detach().numpy())
        sum_Forrow = torch.sum(multi, dim=1)
        if reduce:
            rce = torch.mean(sum_Forrow, dim=0)
        else:
            rce = sum_Forrow
        loss_rce = (-1) * rce
        loss = 0.01 * loss_ce.detach() + loss_rce
        return loss.to(args.device)

    def update_weights(self, global_ep, optimizer, epoch, mod_name):
        model = self.model_C
        if args.method == 'SCE':
            criterion = self.sce_loss
        else:
            criterion = nn.CrossEntropyLoss().to(args.device)
        epochs = args.train_epochs_C

        model.train()
        num_iter = len(self.data_loader)
        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels).mean()
            if args.method=='fedprox':
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model_C.parameters()):
                    fed_prox_reg += ((1 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg.to(args.device)
            loss.backward()
            optimizer.step()
            sys.stdout.write('\r')
            sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                             % (self.client_id, global_ep, epoch, epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()
            del inputs, labels, outputs, loss
            if args.machine:
                ct.empty_cached_memory()
            else:
                torch.cuda.empty_cache()

    def local_update(self, epoch, mod_name):

        model = self.model_C
        args.dlr = 0.001
        optimizer = optim.SGD(model.parameters(), lr=args.dlr, momentum=args.mom, weight_decay=args.decay)
        iterations = args.train_epochs_C
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

        for iter in range(iterations):
            self.update_weights(epoch, optimizer, iter, mod_name)
            scheduler.step()
          

        return model.state_dict(), len(self.dataset)

class Baselines:
    def __init__(self):
        if args.prepare_data:
            self.prepare_data()
        self.clients = [[] for i in range(args.num_users)]
        self.ini_model()
        for p_id in range(args.num_users):
            self.clients[p_id] = Client(p_id, copy.deepcopy(self.model_C), copy.deepcopy(self.model_B), clientPath)
        self.server = Server(self.model_C, self.model_B, serverPath)
    
        if args.autoload:
            torchLoad = torch.load( os.path.join(checkpoint_dir, 'servermodel_'+str(args.start_epoch)+'.json'))
            self.server.model_C.load_state_dict(torchLoad['netC_state_dict'])

        # args.start_epoch = args.start_epoch +1
        self.server.sendmodel()
        for p_id in range(args.num_users):
            self.clients[p_id].receivemodel()

    def create_model(self):
        model = ResNet18(num_classes=args.num_classes)
        model = model.to(args.device)
        return model

    def ini_model(self):
        if args.dataset == 'cifar10':
            self.model_C = self.create_model()
            self.model_B = self.create_model()

    def make_noise(self, oriset, noise='SVHN', ix=0):

        uni_nm = len(oriset)
        num_all_noise = int(uni_nm * args.r)
        num_open_noise = int(num_all_noise * args.on)
        if noise == 'cifar100':
            noise_data = unpickle('data/cifar-100/cifar-100-python/train')['data']
            noise_data = noise_data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        elif noise == 'SVHN':
            noise_data = sio.loadmat('data/SVHN/train_32x32.mat')
            noise_data = noise_data['X']
            noise_data = noise_data.transpose((3, 0, 1, 2))

        labels = copy.deepcopy(oriset.labels)
        images = oriset.samples
        idx = list(range(uni_nm))
        target_noise_idx = list(range(uni_nm))
        random.shuffle(target_noise_idx)
        # ========asymmetric noise=========
        if args.mode == 'asym':
            cr = args.r - args.r * args.on
            closed_idx = []
            noise_mapping = [2, 9, 0, 5, 7, 3, 6, 4, 8, 1]
            noise_matrix = np.zeros((args.num_classes, args.num_classes), dtype=float)
            for i in range(args.num_classes):
                if i == noise_mapping[i]:
                    noise_matrix[i][i] = 1
                else:
                    for j in range(args.num_classes):
                        if j == noise_mapping[i]:
                            noise_matrix[i][j] = cr
                        else:
                            if j == i:
                                noise_matrix[i][j] = 1 - cr
                            else:
                                noise_matrix[i][j] = 0
            print('noise_matrix:')
            print(noise_matrix)
            for i in range(len(labels)):
                labels[i] = np.random.choice(np.arange(args.num_classes), 1, p=noise_matrix[labels[i]])
                if labels[i] != oriset.labels[i]:
                    closed_idx.append(i)
            open_idx = set(idx) - set(closed_idx)
            open_idx = list(open_idx)
            random.shuffle(open_idx)
            open_idx = open_idx[:num_open_noise]
        else:
            # ========symmetric noise=========
            random.shuffle(idx)
            open_idx = idx[:num_open_noise]
            closed_idx = idx[num_open_noise:num_all_noise]
            for i in range(uni_nm):
                if i in closed_idx:
                    rag = list(range(args.num_classes))
                    rag.remove(labels[i])
                    labels[i] = random.choice(rag)
        #
        # self.clients[ix].conf_mat(debug1,debug)
        open_map = list(zip(open_idx, target_noise_idx[:num_open_noise]))
        for cleanIdx, noisyIdx in open_map:
            images[cleanIdx] = noise_data[noisyIdx]

        global noise_chart_op, noise_chart_cl
        noise_chart_op[ix] = [True if i in open_idx else False for i in range(uni_nm)]
        noise_chart_op[ix] = np.array(noise_chart_op[ix])
        noise_chart_cl[ix] = [True if i in closed_idx else False for i in range(uni_nm)]
        noise_chart_cl[ix] = np.array(noise_chart_cl[ix])
        newset = data_loader(images, labels, args)
        return newset

    def bench_left(self, train_dataset):
        bench_idxs = bench_assign(train_dataset)
        X, Y = train_dataset.samples[bench_idxs], train_dataset.labels[bench_idxs]
        idxs_all = list(range(len(train_dataset)))
        idxs_left = list(set(idxs_all) - set(bench_idxs))
        X_left, Y_left = train_dataset.samples[np.array(idxs_left)], train_dataset.labels[np.array(idxs_left)]
        left_dataset = data_loader(X_left, Y_left, args)
        bench_dataset = data_loader(X, Y, args)
        return left_dataset, bench_dataset

    def prepare_data(self):
        if args.dataset == 'cifar10':
            train_dataset, test_dataset = get_dataset(args)

            left_dataset, bench_dataset = self.bench_left(train_dataset)
            Usr_dataset = get_Users_Data(args, left_dataset)

        elif args.dataset == 'camelyon17':
            Usr_dataset, test_dataset = get_Users_Data(args)
            test_dataset, bench_dataset = self.bench_left(test_dataset)
        # new_dataset = self.make_noise(left_dataset)
        # global noise_chart_cl, noise_chart_op
        for p_id in range(args.num_users):
            with open(os.path.join(clientPath, 'clean_labels' + str(p_id) + '.pkl'), 'wb') as f:
                pickle.dump(Usr_dataset[p_id].labels, f)

        for i in range(args.num_users):
            if noise_idx[i]:
                Usr_dataset[i] = self.make_noise(Usr_dataset[i], noise=args.noisy_dataset, ix=i)

        Path(clientPath).mkdir(parents=True, exist_ok=True)
        Path(serverPath).mkdir(parents=True, exist_ok=True)
        for p_id in range(args.num_users):
            with open(os.path.join(clientPath, 'dataset' + str(p_id) + '.pkl'), 'wb') as f:
                pickle.dump(Usr_dataset[p_id], f)
        with open(os.path.join(serverPath, 'testdata.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)
        with open(os.path.join(serverPath, 'benchdata.pkl'), 'wb') as f:
            pickle.dump(bench_dataset, f)
        with open(os.path.join(serverPath, 'noise_chart_op.pkl'), 'wb') as f:
            pickle.dump(noise_chart_op, f)
        with open(os.path.join(serverPath, 'noise_chart_clo.pkl'), 'wb') as f:
            pickle.dump(noise_chart_cl, f)

    def compute_lamda(self, samp1, samp2):

        # samp1 = (samp1-samp1.min())/(samp1.max()-samp1.min())
        # samp2 = (samp2-samp2.min())/(samp2.max()-samp2.min())
        user_len = len(samp2)
        # samp2.squeeze()
        enu1 = sorted(enumerate(samp1), key=itemgetter(1))
        enu2 = sorted(enumerate(samp2), key=itemgetter(1))

        sp1 = [value for index, value in enu1]
        sp2 = [value for index, value in enu2]

        temp = -1

        D = stats.ks_2samp

        inixis = D(samp1, samp2)
        inixis = inixis[0]

        distance = []
        stp = 100
        STEP = np.arange(user_len / stp, user_len + 1, user_len / stp)
        STEP = STEP.astype(np.int)
        STEP_loss = []
        idx = 0
        # for t in range(1,len(samp2)):
        for t in STEP:
            trunsp2 = sp2[:t]
            STEP_loss.append(sp2[t - 1].item())
            dis = D(sp1, trunsp2)
            dis = dis[0]
            distance.append(dis)
            if dis <= inixis:
                inixis = dis
                temp = t - 1
                best_idx = idx
            idx += 1
        lamda = sp2[temp]

        kneedle_cov_dec2 = KneeLocator(STEP, distance, curve='convex', direction='decreasing',
                                       online=True)
        kneedle_cov_dec2.plot_knee()
        plt.savefig('DS_kneeKS_0929.png')
        with open(os.path.join(serverPath, 'lamda.pkl'), 'wb') as f:
            pickle.dump(lamda, f)
        return lamda

    def Data_selection(self, max_epochs=200):

        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]
        for epoch in range(args.start_epoch,max_epochs):
            if epoch == args.start_epoch:
                if not args.load_loss:
                    bench_loss = self.server.cmp_bench_loss()
                    uni_loss = []
                    for ix in range(args.num_users):
                        _, eachloss = self.clients[ix].reference()
                        with open(os.path.join(clientPath,'DS_loss_%s'%(str(ix))),'wb') as f:
                            pickle.dump(eachloss,f)
                        uni_loss.extend(eachloss.tolist())
                        del _,eachloss
                        if args.machine:
                            ct.empty_cached_memory()
                        else:
                            torch.cuda.empty_cache()
                    uni_loss = np.array(uni_loss)
                    # uni_loss = (uni_loss-uni_loss.min())/(uni_loss/max()-uni_loss.min())
                    lamda = self.compute_lamda(bench_loss, uni_loss)
                    with open(os.path.join(clientPath,'DS_lamda'),'wb') as f:
                        pickle.dump(lamda,f)
                else:
                    with open(os.path.join(clientPath,'DS_lamda'),'rb') as f:
                        lamda = pickle.load(f)
                    for ix in range(args.num_users):
                        with open(os.path.join(clientPath,'DS_loss_%s'%(str(ix))),'rb') as f:
                            self.clients[ix].local_loss = pickle.load(f)
                for ix in range(args.num_users):
                    self.clients[ix].lamda1_detect(lamda)

            for ix in range(args.num_users):
                model_param[ix], Keep_size[ix] = self.clients[ix].local_update(epoch, 'netD')

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch, 'netD')

            for ix in range(args.num_users):
                self.clients[ix].receivemodel()
            if epoch % 10 == 0:
                torch.save(self.server.model_C.state_dict(), os.path.join(checkpoint_dir,
                                                                          'global_train_%d.json' % (epoch)))
            if self.server.stop:
                break

    def logDetectIndex(self, recalls, precisions, f1scores, epoch):

        writer.add_scalars('Attacker_Index/clean',
                           {'recall': recalls[0], 'precision': precisions[0],
                            'f1score': f1scores[0]}, epoch)
        writer.add_scalars('Attacker_Index/open',
                           {'recall': recalls[1], 'precision': precisions[1],
                            'f1score': f1scores[1]}, epoch)
        writer.add_scalars('Attacker_Index/close',
                           {'recall': recalls[2], 'precision': precisions[2],
                            'f1score': f1scores[2]}, epoch)

    def FL(self):
        weights_C = [[] for i in range(args.num_users)]
        Keep_size_D = [0] * args.num_users
        global global_weight_collector,clnt_selec
        incenti_scheme = ['sfavg','soft_sfavg','sfavg_all','labUnion','labUnionSel','individual','individualSel','FLI','FLISel']
        for epoch in range(args.start_epoch, args.num_epochs + 1):
            if args.method in incenti_scheme:
                if 'labUnion' in args.method:
                    avg_func = self.server.LabUnion
                elif 'individual' in args.method:
                    avg_func = self.server.Individual
                elif 'FLI' in args.method:
                    avg_func = self.server.FLI
                else:
                    avg_func = self.server.Sfavg
                writer.add_scalars(args.method + '/sv',
                                   {'sv_0': sv[0], 'sv_1': sv[1], 'sv_2': sv[2], 'sv_3': sv[3]}, epoch)
                writer.add_scalars(args.method + '/rev_score',
                                   {'rev_0': rev[0], 'rev_1': rev[1], 'rev_2': rev[2], 'rev_3': rev[3]}, epoch)
                writer.add_scalars(args.method + '/par',
                                   {'par_0': par[0], 'par_1': par[1], 'par_2': par[2], 'par_3': par[3]}, epoch)
                if args.method == 'sfavg' or 'Sel' in args.method:
                    clnt_selec = np.random.choice(range(args.num_users),args.m,replace=False,p=par)
                    clnt_selec = list(clnt_selec)
                else:
                    clnt_selec = range(args.num_users)

            else:
                clnt_selec = range(args.num_users)
                avg_func = self.server.FedAvg
            global_weight_collector = list(self.server.model_C.parameters())
            for ix in clnt_selec:
                print('\nTrain netC')
                weights_C[ix], Keep_size_D[ix] = self.clients[ix].local_update(epoch, 'netD')

            print('Global model of epoch={} aggregating...'.format(epoch))

            avg_func(weights_C, Keep_size_D)

            if epoch % 20 == 0:
                torch.save({
                    'netC_state_dict': self.model_C.state_dict(),
                }, os.path.join(checkpoint_dir, 'servermodel_20.json'))
            self.server.sendmodel()
            self.server.test(epoch)

            for i in range(args.num_users):
                self.clients[i].receivemodel()
            if self.server.stop:
                break


         
def setup_seed(seed):
    torch.manual_seed(seed)
    if not args.machine:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def img_augmentation(inputs):
    im_aug = tfs.Compose([

        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    for i in range(len(inputs)):
        inputs[i] = im_aug(inputs[i].cpu())

    return inputs

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total,used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_mem(cuda_device):
    # total,used = check_mem(cuda_device)
    # total = int(total)
    # used = int(used)
    # max_mem = int(total*0.9)
    block_mem = 1000
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    print('===========cliam GPU mem for '+str(block_mem)+'Mib==========')
    del x

if __name__ == '__main__':
    # occupy_mem(cuda_device)

    setup_seed(42)

    bl = Baselines()
    if args.method != 'DS':
        bl.FL()
    else :
        bl.Data_selection()

  


