# import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from random import random, choice
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import torchvision
import os
from scipy.ndimage.filters import gaussian_filter
import copy
import torch
from scipy import fftpack
import imageio
from skimage.transform import resize
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


def loadpathslist(root,flag,opt):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            if class_name =='stylegan':
                for object_class in ['bedroom','car','cat']:
                    imgpaths = os.listdir(root+'/'+class_name +'/'+object_class+'/'+flag+'/')
                    for imgpath in imgpaths:
                        paths.append(root+'/'+class_name +'/'+object_class+'/'+flag+'/'+imgpath)
            elif class_name=='stylegan2':
                for object_class in ['horse','church','car','cat']:
                    imgpaths = os.listdir(root+'/'+class_name +'/'+object_class+'/'+flag+'/')
                    for imgpath in imgpaths:
                        paths.append(root+'/'+class_name +'/'+object_class+'/'+flag+'/'+imgpath)
            elif class_name == 'cyclegan':
                for object_class in ['apple','horse','orange','summer','winter','zebra']:
                    imgpaths = os.listdir(root+'/'+class_name +'/'+object_class+'/'+flag+'/')
                    for imgpath in imgpaths:
                        paths.append(root+'/'+class_name +'/'+object_class+'/'+flag+'/'+imgpath)
            elif class_name == 'progan':
                for object_class in ['airplane', 'bird', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'cow', 'chair', 'diningtable', 'dog', 'person', 'pottedplant', 'motorbike', 'tvmonitor', 'train', 'sheep', 'sofa', 'horse']:
                    imgpaths = os.listdir(root+'/'+class_name +'/'+object_class+'/'+flag+'/')
                    for imgpath in imgpaths:
                        paths.append(root+'/'+class_name +'/'+object_class+'/'+flag+'/'+imgpath)
            else:
                imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
                for imgpath in imgpaths:
                    paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths



class read_data_new_cam(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real',opt)
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake',opt)
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        if opt.isTrain:
            crop_func = transforms.RandomCrop(224)
            flip_func = transforms.RandomHorizontalFlip()
        elif opt.isVal:
            crop_func = transforms.RandomCrop(224)
            flip_func = transforms.Lambda(lambda img: img)

        # import pdb
        # pdb.set_trace()
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: data_augment(img, opt) if opt.isTrain else img),
                crop_func,
                # flip_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN['clip'], std=STD['clip'] ),
            ])
        self.transform2 = transforms.Compose([
                transforms.Lambda(lambda img: data_augment(img, opt) if opt.isTrain else img),
                crop_func,
                # flip_func,
                transforms.ToTensor(),
                # transforms.Normalize( mean=MEAN['clip'], std=STD['clip'] ),
            ])

    def __getitem__(self, index):
        name=self.img[index].split('/')[-1]
        name=name.split('.')[0]+'.png'
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]#256,256,3

        imgname = self.img[index]
        img1 = self.transform(img)
        img2 = self.transform2(img)

        return img1, img2, target,name

    def __len__(self):
        return len(self.label)
