import argparse
import os
import util
import torch
#import models
#import data

class TrainOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--isTrain', type=bool, default=True)
        parser.add_argument('--isVal', type=bool, default=False)

       
        parser.add_argument('--local_rank', type=int,default=0)
        parser.add_argument('--dist_url', type=str,default='127.0.0.1:29500')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
        parser.add_argument('--distribute', type=str,default='no', help='choose the detection method')
        parser.add_argument('--device', type=str,default='cuda:0', help='choose the detection method')


    
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', default='0.0,3.0')
        parser.add_argument('--jpg_prob', type=float, default=0.5)
        parser.add_argument('--jpg_method', default='cv2,pil')
        parser.add_argument('--jpg_qual', default='30,100')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--num_threads', type=int, default=200, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--detect_method', type=str,default='SSD', help='choose the detection method')
        parser.add_argument('--dataroot', default='/workspace/datasets/progan', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--classes', default='airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse', help='image classes to train on')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--checkpoints_dir', type=str, default='./ckpts', help='models are saved here')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',help='use pre-trained model')


        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default= 0.00001, help='initial learning rate for adam')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='loss weight for l2 reg')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')


        parser.add_argument('--lr-method', default='step', type=str,help='method of learning rate')
        parser.add_argument('--lr-params', default=[10,20,30], dest='lr_params',nargs='*',type=float,action='append', help='params of lr method')
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = True   # train or test
        opt.isVal = False
        opt.classes = opt.classes.split(',')

        # result dir, save results and opt
        opt.results_dir=f"./results/{opt.detect_method}"
        util.mkdir(opt.results_dir)



        if print_options:
            self.print_options(opt)



        # additional
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt

class TestOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--isTrain', type=bool, default=False)
        parser.add_argument('--isVal', type=bool, default=True)
        parser.add_argument('--isType', type=bool, default=False)
        parser.add_argument('--type',type=str,default='ADM')
        parser.add_argument('--distribute', type=str,default='no', help='choose the detection method')
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--noise_type', type=str,default=None, help='such as jpg, blur and resize')
        parser.add_argument('--blur_sig', default='1.0')
        parser.add_argument('--jpg_method', default='pil')
        parser.add_argument('--jpg_qual', default='95')
        parser.add_argument('--local_rank', type=int,default=0)
  
        parser.add_argument('--dataroot', default='/workspace/datasets/progan/val', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        parser.add_argument('--batch_size', type=int, default=30, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--model_path',type=str,default='./ckpts/SSDmodel_epoch_best.pth',help='the path of detection model')
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--device', type=str,default='cuda:0', help='choose the detection method')
        parser.add_argument('--detect_method', type=str,default='SSD', help='choose the detection method')
        self.initialized = True

        return parser

    def gather_options(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        file_name = os.path.join(opt.results_dir, f'{opt.noise_type}opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        # opt.isTrain = False   # train or test
        # opt.isVal = False

        # result dir, save results and opt
        opt.results_dir=f"./results/{opt.detect_method}"
        util.mkdir(opt.results_dir)



        if print_options:
            self.print_options(opt)



        # additional

        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
