import functools
import torch
import torch.nn as nn

from .base_model import BaseModel, init_weights

from util import get_model
from .SSDLoss import MultiLoss

'''
trainer：
包含模型的调用
学习率
输入的图像
模型的前向过程：主要是将输入到模型中得到输出
损失函数
优化参数backward
'''
class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt=opt

        if self.isTrain and not opt.continue_train:
            # 会要用预训练的模型
            self.model = get_model(opt)
            # print('isTrain')


        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)
            # print('continue_train')


        if self.isTrain:

            if opt.detect_method=='SSD':
                self.loss_fn=MultiLoss()
            elif opt.detect_method =='FakeInversion':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            '''
            self.loss_fn = nn.BCEWithLogitsLoss()
            '''
            # initialize optimiz ers
            if opt.detect_method =='FakeInversion' :
                params = self.model.classifier.parameters()
            else:
                params = self.model.parameters()
            #修改fix_backbone
            if self.opt.detect_method == "UnivFD" and self.opt.fix_backbone:
                params = []

                for name, p in self.model.named_parameters():#将模型的全连接层提取出来

                    #if  name=="fc.weight" or name=="fc.bias":
                    #    params.append(p)
                    if ('encoder' in name) or name=='fc1.weight' or name=='fc1.bias':
                        params.append(p)
                    else:
                        p.requires_grad = False
            if opt.optim == 'adam':
                if self.opt.detect_method == "UnivFD" or self.opt.detect_method == "SSD":
                    self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
                else:
                    self.optimizer = torch.optim.Adam(params,lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':

                if opt.pretrained:
                    params_list = [{'params': self.model.features.parameters(), 'lr': opt.lr,
                        'weight_decay': opt.weight_decay},]
                    params_list.append({'params': self.model.representation.parameters(), 'lr': opt.lr,
                        'weight_decay': opt.weight_decay})
                    params_list.append({'params': self.model.classifier.parameters(),
                            'lr': opt.lr*opt.classifier_factor,
                            'weight_decay': 0. if opt.arch.startswith('vgg') else opt.weight_decay})
                else:
                    params_list = [{'params': self.model.features.parameters(), 'lr': opt.lr,
                        'weight_decay': opt.weight_decay},]
                    params_list.append({'params': self.model.representation.parameters(), 'lr': opt.lr,
                        'weight_decay': opt.weight_decay})
                    params_list.append({'params': self.model.classifier.parameters(),
                            'lr': opt.lr*opt.classifier_factor,
                            'weight_decay':opt.weight_decay})
                self.optimizer = torch.optim.SGD(params_list,
                                                 lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        if opt.detect_method=='FakeInversion':
            self.model.classifier.to(self.device)
        else:
            self.model.to(self.device)
        if self.opt.distribute=='yes':
            self.model=torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.local_rank],output_device=self.local_rank,find_unused_parameters=True)

        print(self.device)

    # def adjust_SVD_learning_rate(self,lr_factor, epoch):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     #lr = opt.lr * (0.1 ** (epoch // 20))
    #     groups = ['features']
    #     groups.append('representation')
    #     groups.append('classifier')
    #     num_group = 0
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] *= lr_factor[epoch]
    #         print('the learning rate is set to {0:.15f} in {1:} part'.format(param_group['lr'], groups[num_group]))
    #         num_group += 1


    def adjust_learning_rate(self, min_lr=1e-6,lr_factor=1,epoch=0):#调整学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True
        '''
        for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 10.
                if param_group['lr'] < min_lr:
                    return False
        return True
        '''

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()



    def forward(self):
        if self.opt.detect_method == "Fusing":
            self.output = self.model(self.input_img, self.cropped_img, self.scale)
        elif self.opt.detect_method == "UnivFD":
            # print(self.input.size())
            self.output = self.model(self.input)
            self.output = self.output.view(-1).unsqueeze(1)#view转变为一维向量 然后再增加一个维度
        elif self.opt.detect_method in ["FatFormer",'FakeInversion']:
            self.output = self.model(self.input)
        else:
            self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        if self.opt.detect_method=='SSD':
            self.forward()

            # output,restoration_feature,feature_clip1=self.output
            # output=output.to(self.device)
            # restoration_feature=restoration_feature.to(self.device)
            # feature_clip1=feature_clip1.to(self.device)
            # self.loss = self.loss_fn(output,restoration_feature,feature_clip1,self.label)
            # torch.autograd.set_detect_anomaly(True)

            # output,final_z,log_jac_det,alpha_p,beta_p=self.output
            # output=output.to(self.device)
            # feature_z=final_z.to(self.device)
            # log_jac_det=log_jac_det.to(self.device)
            # alpha_p=alpha_p.to(self.device)
            # beta_p=beta_p.to(self.device)
            # self.loss=self.loss_fn(output,final_z,log_jac_det,alpha_p,beta_p,self.label,self.device)[0]
            #ori对应的代码

            output,final_z,restoration_feature,ori_feature=self.output
            output=output.to(self.device)
            feature_z=final_z.to(self.device)
            restoration_feature=restoration_feature.to(self.device)
            ori_feature=ori_feature.to(self.device)
            self.loss=self.loss_fn(output,feature_z,restoration_feature,ori_feature,self.label)

            # output=self.output
            # output=output.to(self.device)
            # self.loss=self.loss_fn(output,self.label)
            '''
            print('out:')
            print(out)
            print('label:')
            print(self.label)
            '''
            #self.loss = self.loss_fn(out.squeeze(1), self.label)
            self.optimizer.zero_grad()
            self.loss[0].backward()
            self.optimizer.step()
        elif self.opt.detect_method in ["UnivFD",'FakeInversion']:
            self.forward()
            self.output=self.output.to(self.input.device)
            '''
            print('out:')
            print(self.output)
            print('label:')
            print(self.label)
            '''
            # print(torch.argmax(torch.softmax(self.output.squeeze(1),dim=1),dim=1))
            # print('shhshshjsjsjjs')
            # print(self.label.long())
            self.loss = self.loss_fn(self.output.squeeze(1), self.label.long())
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        else:
            self.forward()
            self.output=self.output.to(self.input.device)
            '''
            print('out:')
            print(self.output)
            print('label:')
            print(self.label)
            '''
            self.loss = self.loss_fn(self.output.squeeze(1), self.label)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

