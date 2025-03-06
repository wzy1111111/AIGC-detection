# from pix2pix
import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler



class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain

        self.save_dir = os.path.join(opt.checkpoints_dir)
        if opt.distribute=='yes':
            local_rank=torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            global device
            device=torch.device('cuda',local_rank)
            self.local_rank=local_rank
        else:
            device=opt.device
            self.local_rank=None
        self.device = device

    def save_networks(self,method, epoch):
        save_filename = method+'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        '''
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }
        '''
        state_dict=self.model.state_dict()
        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, method,epoch):
        load_filename = method+'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)

        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.model.load_state_dict(state_dict)




    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
