from .lora_clip.loraclip import lora_clip as loraclip
from .lora_clip import loraclip as lorautil
import torch
import torch.nn as nn
#from base_model import BaseModel, init_weights
import torch.nn.functional as F
import numpy as np
from glob import glob
# from .univfd_models.clip import clip 
# from .resnet import resnet50
from collections import OrderedDict
from typing import Tuple, Union
from .clip_models import *
from .clip import clip
# from .spectformer import *




class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.k_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim_k = k.shape#N 257 1024 1 257 256
        bsz_q, tgt_len_q, embed_dim_q = q.shape
        query_states = self.q_proj(q) * self.scale

        key_states = self._shape(self.k_proj(k), -1, bsz)# b h 257 1024//h
        value_states = self._shape(self.v_proj(v), -1, bsz)#b h 257 1024//h

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)# N*h 257 1024//h
        
        query_states = self._shape(query_states.expand(bsz,tgt_len_q,self.embed_dim), tgt_len_q, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)


        src_len = key_states.size(1)#257
        
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))#B*h q_len k_len

        if attn_weights.size() != (bsz * self.num_heads, tgt_len_q, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range
        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:

            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)#
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len_q, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len_q, self.head_dim)#b n q_len h 
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len_q, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output.clone().detach().requires_grad_(True)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias

class Encoder(nn.Module):
    def __init__(self,q_dim=1024,k_dim=1088,v_dim=1088,embed_dim=2176, num_heads=16):
        super().__init__()
        self.norm1=nn.LayerNorm(k_dim)
        self.norm2=nn.LayerNorm(k_dim)
        self.norm_1=nn.LayerNorm(k_dim)
        self.norm_1_1=nn.LayerNorm(k_dim)
        self.multiHeadAttention1=MultiHeadAttention(q_dim=q_dim,k_dim=k_dim,embed_dim=embed_dim, num_heads=num_heads)
        #1024 1088 2716 
        self.multiHeadAttention2=MultiHeadAttention(q_dim=k_dim,k_dim=k_dim,embed_dim=embed_dim, num_heads=num_heads)
    def forward(self,feature_in: torch.Tensor,feature_bags:torch.tensor):
        feature_in=self.norm1(feature_in)
        feature_in1=self.norm_1(self.multiHeadAttention1(self.norm2(feature_bags),feature_in,feature_in))#B len_q d_v B 5345 1088

        feature_in2=self.norm_1_1(self.multiHeadAttention2(feature_in,feature_in1,feature_in1)+feature_in)#B 257 1024  B 257 1088
        return feature_in1,feature_in2

class Decoder(nn.Module):
    def __init__(self,q_dim=1024,k_dim=1088,v_dim=1088,embed_dim=2176, num_heads=16):
        super().__init__()
        self.norm_2=nn.LayerNorm(k_dim)
        self.norm_2_2=nn.LayerNorm(k_dim)
        #decoder
        self.multiHeadAttention_1=MultiHeadAttention(q_dim=k_dim,k_dim=k_dim,embed_dim=embed_dim, num_heads=num_heads)
        #1024 1088 2716 
        self.multiHeadAttention_2=MultiHeadAttention(q_dim=k_dim,k_dim=k_dim,embed_dim=embed_dim, num_heads=num_heads)
    def forward(self,feature_in1: torch.Tensor,feature_in2:torch.tensor):
        feature_out_1=self.norm_2(self.multiHeadAttention_1(feature_in2,feature_in2,feature_in2)+feature_in2)
        feature_out_2=self.norm_2_2(self.multiHeadAttention_2(feature_out_1,feature_in1,feature_in1)+feature_out_1)
        return feature_out_2

class LowFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=9, stride=7, padding=1)#16->128
        self.bn_de1 = nn.BatchNorm2d(128)
        
        self.deconv2=nn.ConvTranspose2d(in_channels=1024, out_channels=64, kernel_size=4, stride=4, padding=4)#16->64
        self.bn_de2 = nn.BatchNorm2d(64)


        self.conv1=nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2=nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1)


    def forward(self,semantic_map:torch.Tensor, x: torch.Tensor):
        B,S,C=semantic_map.size()
        semantic_map=semantic_map.view(B, 16, 16, C).permute(0,3,1,2)

        
        x_1=F.relu(self.bn1(self.conv1(x)))# B 112 112 128 B H W C 
        s_1=F.relu(self.bn_de1(self.deconv1(semantic_map)))#B 112 112 C=>128
        x_s_1=torch.mul(x_1, s_1)
        x_c_1=F.relu((torch.abs(x_s_1-x_1))*x_1+x_1)

        x_2=F.relu(self.bn2(self.conv2(x_c_1)))# B 56 56 64
        s_2=F.relu(self.bn_de2(self.deconv2(semantic_map)))#B 56 56 C=>64
        x_s_2=torch.mul(x_2, s_2)
        x_c_2=F.relu((torch.abs(x_s_2-x_2))*x_2+x_2)

        x_3=F.relu(self.bn3(self.conv3(x_c_2)))# B 28 28 1

        return x_3.squeeze(-1).view(B, -1), [x_3, x_c_2, x_c_1]


class GCP(nn.Module):

    def __init__(self,q_dim=1024,k_dim=1088,v_dim=1088,embed_dim=2176, num_heads=16,conditional_gate=True,out_dim=256,
                 drop_path=.0, init_values=1e-4, ):
        super().__init__()
        self.q_dim=q_dim

        #encoder
        self.encoder=Encoder(q_dim=q_dim,k_dim=k_dim,embed_dim=embed_dim, num_heads=num_heads)
        #1024 1088 2716 

        #decoder
        self.decoder=Decoder(q_dim=q_dim,k_dim=k_dim,embed_dim=embed_dim, num_heads=num_heads)

        self.lowFeature=LowFeature()
        self.linear=nn.Linear(784,256)

    def forward(self, x:torch.Tensor,feature_in: torch.Tensor,feature_bags:torch.tensor):
        #feature_out B 256 256

        #encoder
        feature_in1,feature_in2=self.encoder(feature_in,feature_bags)

        #decoder
        feature_out_1=self.decoder(feature_in1,feature_in2)#B 257 1024

        difference_part=torch.abs((feature_out_1-feature_in)[:,1:,:])

        x_3, x_3_ori = self.lowFeature(difference_part,x)
        
        extract_feature=self.linear(x_3)
 
        return feature_out_1,extract_feature, x_3_ori


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class PEM(nn.Module):
    def __init__(self,dim=1024):
        super(PEM, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dim=dim
    def forward(self,prior_features:torch.Tensor,feature_clip:torch.Tensor):
        shape=feature_clip.shape
        x1=self.norm1(prior_features).view(-1, self.dim).unsqueeze(0)
        x2=self.norm2(feature_clip).view(-1, self.dim).unsqueeze(1)
        similarity=F.cosine_similarity(x1,x2,dim=2)

        _, top_indices = torch.topk(similarity, k=64, dim=1,largest=True)

        return top_indices/int(prior_features.shape[1])


class BiGFF(nn.Module):
    '''Bi-directional Gated Feature Fusion.'''
    
    def __init__(self,d_model):
        super(BiGFF, self).__init__()

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

        self.frequency_gamma = nn.Parameter(torch.zeros(1))
        self.normal_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, frequency_feature, normal_feature):

        energy = torch.cat((frequency_feature, normal_feature), dim=1)

        gate_structure_to_frequency =  self.linear1(frequency_feature)
        gate_texture_to_normal = self.linear2(normal_feature)

 
        frequency_feature = frequency_feature + self.frequency_gamma * (gate_structure_to_frequency * normal_feature)
        normal_feature = normal_feature + self.normal_gamma * (gate_texture_to_normal * frequency_feature)

        return torch.cat((frequency_feature, normal_feature), dim=2)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature   
        return out, x 



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
       

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x 


class SSD(nn.Module):
    # 这是网络结构
    def __init__(self,name='ViT-L/14',num_classes=1):
        super(SSD, self).__init__()
        width1=256

        width2=1280

        self.fc1 = nn.Linear(width2,256)
        self.ac=nn.ReLU()
        self.fc=nn.Linear(256,1)

        self.gcp=GCP(q_dim=1024,k_dim=1024,v_dim=1024,embed_dim=2176, num_heads=16,conditional_gate=True,out_dim=width1)
        self.BN1=nn.BatchNorm1d(width2)
        self.BN2=nn.BatchNorm1d(256)
        

        self.clip_model_name=name
        self.device='cpu'
        self.lora_rank=6 
        self.lora_alpha=6
        self.lora_dropout=0.05
        self.lora_mode='vision'   
        self.clip,_ = loraclip.load(self.clip_model_name, self.device, r=self.lora_rank,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout,lora_mode=self.lora_mode)

    def forward(self, x,feature_bags=None):
        
     
        feature_clip1 =self.clip.encode_image(x)[:,0,:]
        feature_clip = self.clip.encode_image(x)
        prior_features=torch.load('../datasets/feature_bags_5000_final.pt').reshape(1,-1,feature_clip.shape[2]).to(x.device)#1 5345 1024
     
        fusion_feature=feature_clip
     
        restoration_feature,final_feature, x_3_ori=self.gcp(x,fusion_feature,prior_features)#B 257 1088   B 256 256
 
        features_fake_final=torch.cat((feature_clip[:,0,:],final_feature),dim=1)
        output=self.fc(self.ac(self.BN2(self.fc1(self.BN1(features_fake_final)))))

        return output,final_feature,restoration_feature,feature_clip, x_3_ori



if __name__ == "__main__":
    x=torch.randn(2,3,224,224)
    model=SSD('ViT-L/14')
    model(x)
    