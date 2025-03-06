import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K,margin_arg):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.margin_arg=margin_arg
        self.fc = Parameter(torch.Tensor(dim, cN*K))
       
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0).to(input.device)
        simInd = input.matmul(centers)#
        simStruc = simInd.reshape(-1, self.cN, self.K)#
        prob = F.softmax(simStruc*self.gamma, dim=2)

        simClass = torch.sum(prob*simStruc, dim=2)#B cn
        marginM = torch.zeros(simClass.shape).cuda()#                                                                                                                                                                           uk

     
        marginM[torch.arange(0, marginM.shape[0]), target.long()] =self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target.long())
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

class MultiLoss(nn.Module):

    def __init__(self):
        super(MultiLoss,self).__init__()
        self.softTriple=SoftTriple(la=20, gamma=0.1, tau=0.2, margin=0.0001, dim=256, cN=2, K=8,margin_arg=0.0001)

    def js_div(self,p_output,q_output,target):
        """
        Function that measures JS divergence between target and output logits:
        """
        kl_div = nn.KLDivLoss(reduction='batchmean')
        target[target==0]=-1
        p_output = F.softmax(p_output,dim=-1).reshape(-1,1024)#3072 1024  12 256 1024
        q_output = F.softmax(q_output,dim=-1).reshape(-1,1024)
        log_mean_output = ((p_output + q_output )/2).log()
        print(p_output.size())
        print(p_output)
        print(p_output*target)
        return 1-(kl_div(log_mean_output*target, p_output*target) +kl_div(log_mean_output*target, q_output*target))/2


    def cosLoss(self,restoration_feature,feature_clip,target):
        B,H,W=restoration_feature.shape
        logit_scale= nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        image_embeddings= feature_clip[:,0,:]
        restoration_embeddings=restoration_feature[:,0,:] 
        cosine_similarities =1- F.cosine_similarity(image_embeddings, restoration_embeddings)
        loss =  sum(cosine_similarities*target)/sum(target)
        return loss

    def forward(self,out_put,final_feature,restoration_feature,ori_feature,target):

        criterion_bce=nn.BCEWithLogitsLoss()
        B,S,C=restoration_feature.size()
        loss_bce= criterion_bce(out_put.squeeze(1),target)*3
        loss_tri=self.softTriple(final_feature, target)/3
        loss_con=torch.norm(restoration_feature - ori_feature, p=2)/(B*S)
        a=0.5
        b=1
        c=1 
        loss=a*loss_bce+b*loss_con+c*loss_tri

        return loss,loss_bce,loss_con,loss_tri

if __name__=='__main__':
    feature_bag=torch.randn(10,267,1024)
    feature_in=torch.randn(1,267,1024)
    out=torch.zeros(1,1)
    target=torch.zeros(1,1)
    out_cons=torch.zeros(1,1)
    model=MultiLoss()
    model(feature_bag,feature_in,out,out_cons,target)
