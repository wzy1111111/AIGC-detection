import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import read_data_new
from .datasets_cam import read_data_new_cam
from torch.utils.data import Dataset
import random

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def get_bal_sampler(dataset):
    targets = []
    for data, label in dataset:
        targets.append(label)

    ratio = np.bincount(np.array(targets))

    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets] 

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)  

    return sampler


def create_dataloader_new(opt):
    shuffle = True if opt.isTrain else False
    dataset = read_data_new(opt)
    if opt.distribute=='yes':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if opt.distribute=='yes':
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=int(16),
                                              pin_memory=False,
                                              sampler=train_sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=int(16)
                                             )
        return data_loader,None
    return data_loader,train_sampler


def create_dataloader_new_cam(opt):
    shuffle = True if opt.isTrain else False
    dataset = read_data_new_cam(opt)
    if opt.distribute=='yes':
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if opt.distribute=='yes':
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=int(16),
                                              pin_memory=False,
                                              sampler=train_sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=int(16)
                                             )
        return data_loader,None
    return data_loader,train_sampler
