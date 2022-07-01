import pylab
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import mpl
import numpy as np
import pandas as pd
import os
import matplotlib
import seaborn as sns
import joblib
from pylab import *
from PIL import Image, ImageDraw,ImageFont,ImageEnhance
import random
import pickle

def eval_byrounds(path,bs='individual',title='rev'):
    """
    :param path: the source file pathdir
    :param bs: choose a scheme from ['sfavg','labUnion','individual','fli']
    :param title: select a mode from ['rev' or 'par']
    :return: relevance score or participation share in each round
    """
    plt.figure(num=1)
    plotHis = []
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    plt.tick_params(labelsize=25)
    flist = os.listdir(path+bs)
    indexes = ['client1', 'client2', 'client3','client4']
    rev_lst = []
    for i in flist:
        if title in i:
            csv = pd.read_csv(os.path.join(path + bs,i))
            df = pd.DataFrame(csv)
            x = df['Step'].array
            x = np.array(x)
            y = df['Value'].array
            y = np.array(y)
            x = x[0:51:5]
            y = y[0:51:5]
            rev_lst.append(y[-1])
            p, = plt.plot(x, y, lw=3,alpha=0.8,marker='o',linestyle='-',markersize=10)
            plotHis.append(p)
    rev_lst = np.array(rev_lst)
    rev_lst = rev_lst/sum(rev_lst)
    plt.ylim((0, 20))
    plt.xlim((0,50))
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    plt.legend(plotHis, indexes,  prop=font1)
    plt.tight_layout()
    plt.savefig(path+title+'_by_rounds.png',bbox_inches='tight')
    plt.show()


def follow_scheme(path,epson=0.1):
    """
    :param path: the source file pathdir
    :param epson: Epson-greedy method, i.e., choose the maximum average payoff with the probability of (1-epson)+epson/2
    :return: probability of following each scheme
    """
    scheme = ['favg','sfavg','labUnion','individual','fli']
    max_pro = 1 - epson + epson / len(scheme)

    fed_val = np.array([0.25]*50)
    cum_fedval = fed_val.cumsum()
    mean_fedval = cum_fedval/np.arange(1,51)
    vote = [[0]*len(scheme) for i in range(4)]

    for i in range(4):
        mean_val_lst = []
        mean_val_lst.append(mean_fedval)
        for inc in range(1,len(scheme)):
            files = os.listdir(path + scheme[inc] + '/')
            df = pd.read_csv(path+scheme[inc]+'/'+files[i])
            val = df['Value'].values
            cum_val = val.cumsum()
            div = df['Step'].values
            mean_val = cum_val/div
            mean_val_lst.append(mean_val)
        for ep in range(50):
            feda = np.array([mean_val_lst[inc][ep] for inc in range(len(scheme))])
            feda = np.argsort(feda)
            chce = [epson/len(scheme)]*(len(scheme)-1)+[max_pro]
            choi = np.random.choice(feda,p=chce)
            vote[i][choi] += 1
        vote[i] = [j*2 for j in vote[i]]

    plt.figure(num=1)
    data = pd.DataFrame(data=np.array(vote), columns=scheme,
                 index=['client 1', 'client 2', 'client 3', 'client 4'])
    data.plot.bar(stacked=True, alpha=0.5)  # 绘制堆积柱状图
    font1 = {'family': 'sans-serief',
             'weight': 'normal',
             'size': 10,
             }
    plt.legend(prop=font1)  # 设置legend刻度值的字体大小
    plt.xticks(rotation=0)
    plt.xlabel('Data owner')
    plt.ylabel('Probability of following each scheme (%)')
    plt.tight_layout()
    plt.savefig(path+'followed scheme for each client.png',bbox_inches='tight')
    plt.show()
    with open(path+'vote.pkl','wb') as f:
        pickle.dump(vote,f)

def data_collection(path):
    """
    :param path: the source file pathdir
    :return: Data Quantity * Quality as a % ahieved by scheme
    """
    with open(path+'vote.pkl','rb') as f:
        vote = pickle.load(f)
    scheme = ['favg', 'sfavg', 'labUnion', 'individual','fli']
    rho = [1,1,1,0.5]
    mean_collection = []
    for i in range(len(scheme)):
        all = 0
        ideal_all = 0
        for j in range(4):
            vote[j][i]/=2
            vote[j][i] = vote[j][i]*rho[j]
            all += vote[j][i]
            ideal_all += 50*rho[j]
        mean = all/ideal_all
        mean *= 100
        mean_collection.append(mean)
    fig = plt.figure()
    plt.bar(scheme,mean_collection,width=0.6,color='m',alpha=0.5)
    plt.xticks(rotation=0)
    plt.xlabel('Incentive mechanism')
    plt.ylabel('Data assets as a % ahieved by scheme')
    plt.tight_layout()
    plt.savefig(path + 'Data assets for each scheme.png', bbox_inches='tight')
    plt.show()
    debug = True

def revenue(path):
    """
    :param path: the source file pathdir
    :return: total revenue as a % ahieved by scheme
    """
    with open(path+'vote.pkl','rb') as f:
        vote = pickle.load(f)
    scheme = ['favg', 'sfavg', 'labUnion', 'individual','fli']
    rho = [1,1,1,0.5]
    reven_collection = []
    for i in range(len(scheme)):
        all = 0
        ideal_all = 0
        for j in range(4):
            vote[j][i]/=2
            vote[j][i] = vote[j][i]*rho[j]
            all += vote[j][i]
            ideal_all += 50  * rho[j]
        reven = np.log(1+all)
        ideal_reven = np.log(1+ideal_all)
        reven /= ideal_reven
        reven *= 100
        reven_collection.append(reven)
    fig = plt.figure()
    plt.bar(scheme,reven_collection,width=0.6,color='m',alpha=0.5)
    plt.xticks(rotation=0)
    plt.xlabel('Incentive mechanism')
    plt.ylabel('Total revenue as a % ahieved by scheme')
    plt.tight_layout()
    plt.savefig(path + 'Revenue for each scheme.png', bbox_inches='tight')
    plt.show()
    debug = True

def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


if __name__=='__main__':
    setup_seed(42)
    incentive_dir = 'tenboCSV/Incentive Mechanism/'
    rcParams.update({'font.size': 14, 'font.family': 'sans-serief',
                     'font.weight': 'normal', 'grid.linewidth': 3})
    eval_byrounds(incentive_dir,title='rev')
    # follow_scheme(incentive_dir)
    # data_collection(incentive_dir)
    # revenue(incentive_dir)
