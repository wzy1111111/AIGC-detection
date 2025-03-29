import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
# from tensorboardX import SummaryWriter
from earlystop import EarlyStopping
import numpy as np

from validate import validate
from data import create_dataloader_new
from network.trainer import Trainer
#from config.gram_options import TrainOptions
from config.ssd_options import TrainOptions 
from util import set_random_seed
from tqdm import tqdm
import matplotlib.pyplot as plt
#from src.network.alexnet import AlexNet
import torch.distributed as dist
from math import inf
from copy import deepcopy
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"





def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.isVal = True
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.batch_size=24

    return val_opt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse(print_options=False)
    val_opt =get_val_opt()

    model = Trainer(opt)
    with open('output1.txt', 'a') as file:
        file.write('other dataset \n')
    root='/workspace/datasets/progan/test1'
    classes =  os.listdir('/workspace/datasets/progan/test1')
    model.eval()
    model.model.load_state_dict(torch.load('./ckpts/SSDmodel_epoch_best.pth'))

    val_opt.dataroot='/workspace/datasets/progan/test1'
    root='/workspace/datasets/progan/test1'
    classes =  os.listdir('/workspace/datasets/progan/test1')
    acc_all =0
    ap_all =0
    auc_all =0
    for objectclass in classes:
        val_opt.dataroot=root+'/'+ objectclass
        acc,ap,auc,f_acc,r_acc= validate(model.model, val_opt)
        acc_all+=acc
        ap_all+=ap
        auc_all+=auc
        with open('output1.txt', 'a') as file:
            file.write('%s :val @  acc: %f  f_acc: %f   r_acc: %f    ap: %f    auc:%f \n' % (objectclass,acc,f_acc,r_acc,ap,auc))

    acc_all=acc_all/19
    ap_all=ap_all/19
    auc_all=auc_all/19
    print('acc_all'+str(acc_all))
    with open('output1.txt', 'a') as file:
        file.write('all val @  acc_all: %f,ap_all: %f,auc_all:%f\n' %  (acc_all,ap_all,auc_all))


