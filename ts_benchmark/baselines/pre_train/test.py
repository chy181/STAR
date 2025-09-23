from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearLoRA(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        rank,
        pre_trained_weight,
        alpha):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 预训练的权重矩阵
        self.pre_trained_weight = pre_trained_weight
        # 冻结，设置为不可训练
        self.pre_trained_weight.weight.requires_grad = False
        self.pre_trained_weight.bias.requires_grad = False

        # 低秩矩阵A和B
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        # 对A矩阵进行高斯初始化
        nn.init.kaiming_normal_(self.lora_A, a = 0.01)

        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = self.alpha / self.rank

    def forward(self, X):
        # X shape :[batch_size, n_vars, patch_num, in_Feature]

        # part 1
        part1 = self.pre_trained_weight(X)
        # part1 shape :[batch_size, n_vars, patch_num, out_features]

        # lora_A lora_B [d, r] [r, d]
        # r [batch_size,patch_num, r, r] b [batch_size,patch_num, d, 1]
        # part 2
        # part2 = X @ self.lora_A @ self.lora_B

        r = []
        b=[]
        # d=out_features
        # lora_A.unsqueeze(0).expand(batch_size*patch_num,-1,-1) 拓展维度
        # 生成W矩阵 lora_A,lora_B,-> [batch_size*patch_num, d,r] [batch_size*patch_num, r,d]  
        # rearange() 换维度
        # r, b -> [batch_size*patch_num, r,r] [batch_size*patch_num, d,1]

        # W = torch.bmm( torch.bmm( lora_A,r), lora_B)*b 
        # W [batch_size*patch_num, in_Feature, out_features]
        # X [batch_size, n_vars, patch_num, in_Feature] 

        # W [batch_size*patch_num, in_Feature, out_features] -> [batch_size, n_vars, patch_num, in_Feature, out_features]
        # X [batch_size, n_vars, patch_num, in_Feature] -> [B, in_Feature] 
        # W [batch_size, n_vars, patch_num, in_Feature, out_features] -> [B, in_Feature, out_features] 
        # X = torch.bmm( X, W) [B, 1 , out_features] 
        # X [batch_size, n_vars, patch_num, out_features]
        part2 = self.scale * part2

        # X [batch_size, n_vars, patch_num, in_Feature, out_features] -> X [B, patch_num, in_Feature, out_features]
        output = part1 + part2
        return output

# 生成r b矩阵的维度变换
# MoE输出 (batch, n_vars, seq_len, d_model)

# (batch, n_vars,patch_num, (patch_len, d_model) ) ->[linear] (batch, n_vars, patch_num, d_model)
    
# Fusion输出 (batch, 1, patch_num, d_model)

# 生成r矩阵 (batch, 1, patch_num, d_model) -> [linear] (batch, 1, patch_num, r*r)
# 生成b矩阵 (batch, 1, patch_num, d_model) -> [linear] (batch, 1, patch_num, outfeatures*1)
    


class LinearLoRA(nn.Module):
    def __init__(self,
        pretrained_linear,
        rank,
        alpha):

        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 预训练的权重矩阵
        self.pre_trained_weight = pretrained_linear
        self.in_features  = self.pre_trained_weight.in_features
        self.out_features  = self.pre_trained_weight.out_features
        # 冻结，设置为不可训练
        self.pre_trained_weight.weight.requires_grad = False
        if self.pre_trained_weight.bias is not None:
            self.pre_trained_weight.bias.requires_grad = False

        # 低秩矩阵A和B
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        # 对A矩阵进行高斯初始化
        nn.init.kaiming_normal_(self.lora_A, a = 0.01)

        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        nn.init.kaiming_normal_(self.lora_B, a = 0.01)
        self.scale = self.alpha / self.rank

        self.r_embed=nn.Linear(self.out_features,rank*rank)
        self.b_embed=nn.Linear(self.out_features,self.out_features*1)

    def forward(self, X, status_token):
        # X shape :[batch_size, n_vars, patch_num, in_Feature]

        # part 1
        part1 = self.pre_trained_weight(X)
        batch_size=part1.shape[0]
        n_vars=part1.shape[1]
        patch_num=part1.shape[2]
        # part1 shape :[batch_size, n_vars, patch_num, out_features]

        # lora_A lora_B [d, r] [r, d]
        # r [batch_size,patch_num, r, r] b [batch_size,patch_num, d, 1]
        # part 2
        # part2 = X @ self.lora_A @ self.lora_B

        r=self.r_embed(status_token) 
        b=self.b_embed(status_token) 
        r=rearrange(r,'batch_size patch_num (r r) -> batch_size patch_num r r')
        b=rearrange(b,'batch_size patch_num (d 1) -> batch_size patch_num d 1')
        # d=out_features
        # 生成W矩阵 lora_A,lora_B,-> [batch_size*patch_num, d,r] [batch_size*patch_num, r,d]  
        self.lora_A.unsqueeze(0).expand(batch_size*patch_num,-1,-1) #拓展维度
        self.lora_B.unsqueeze(0).expand(batch_size*patch_num,-1,-1)
        
        # rearange() 换维度
        # r, b -> [batch_size*patch_num, r,r] [batch_size*patch_num, d,1]
        r=rearrange(r,'batch_size patch_num r r -> (batch_size patch_num) r r')
        b=rearrange(b,'batch_size patch_num d 1 -> (batch_size patch_num) d 1')
        W = torch.bmm( torch.bmm( self.lora_A,r), self.lora_B)*b 
        # W [batch_size*patch_num, in_Feature, out_features]
        # X [batch_size, n_vars, patch_num, in_Feature] 

        # W [batch_size*patch_num, in_Feature, out_features] -> [batch_size, n_vars, patch_num, in_Feature, out_features]
        W = rearrange(W,'(batch_size patch_num) in_Feature, out_features -> batch_size patch_num in_Feature, out_features')
        W.unsqueeze(1).expand(-1,n_vars,-1,-1,-1)
        # X [batch_size, n_vars, patch_num, in_Feature] -> [B, in_Feature] 
        # W [batch_size, n_vars, patch_num, in_Feature, out_features] -> [B, in_Feature, out_features] 
        X=rearrange(X,'b n patch_num in_Feature -> (b n patch_num) in_Feature')
        W=rearrange(W,'b n patch_num in_Feature out_features -> (b n patch_num) in_Feature out_features')
        X = torch.bmm( X, W) #[B, 1 , out_features] 
        # X [batch_size, n_vars, patch_num, out_features]
        
        part2 = self.scale * X

        # X [batch_size, n_vars, patch_num, in_Feature, out_features] -> X [B, patch_num, in_Feature, out_features]
        output = part1 + part2
        return output
