import torch
from torch import nn
from einops import rearrange

class LinearLoRA(nn.Module):
    def __init__(self,
        pretrained_linear,
        wrapper_ref,
        name=None,
        n_continuous=0):

        super().__init__()
        self.pre_trained_weight = pretrained_linear
        self.in_features  = self.pre_trained_weight.in_features
        self.out_features  = self.pre_trained_weight.out_features
        # self.rank = wrapper_ref.config.rank

        self.rank = self.in_features // wrapper_ref.config.rank
        self.alpha = wrapper_ref.config.alpha

        self.name = name
        # 预训练的权重矩阵
        self.n_continuous = n_continuous
        self.n_vars = 1 
        # 冻结，设置为不可训练
        self.pre_trained_weight.weight.requires_grad = False
        if self.pre_trained_weight.bias is not None:
            self.pre_trained_weight.bias.requires_grad = False

        # 低秩矩阵A和B
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank),requires_grad=False)
        # 对A矩阵进行高斯初始化
        nn.init.kaiming_normal_(self.lora_A, a = 0.01)

        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features),requires_grad=False)
        nn.init.kaiming_normal_(self.lora_B, a = 0.01)
        self.scale = self.alpha / self.rank

        self.r_embed=nn.Linear(wrapper_ref.config.d_model,self.rank*self.rank)
        self.b_embed=nn.Linear(wrapper_ref.config.d_model,self.out_features*1)
        self.mlp = nn.Sequential(nn.Linear(wrapper_ref.config.d_model, wrapper_ref.config.d_model), nn.ReLU(), nn.Linear(wrapper_ref.config.d_model, 2*self.rank+2))
        self.wrapper_ref = {'ref':wrapper_ref}
        
    def forward(self, X, tau=0.001):        
        # self.name 
        original_shape = X.shape
        if len(original_shape)==3:
            X = rearrange(X,'(b n) p d -> b n p d', n=self.n_continuous)

        # X shape :[batch_size, n_vars, patch_num, in_Feature]
        # part 1
        part1 = self.pre_trained_weight(X)
        batch_size=X.shape[0]
        self.n_vars=X.shape[1]
        input_patch_num=X.shape[2]

        status_token = self.wrapper_ref['ref'].discrete_token
        patch_num = status_token.shape[2]
        # print(self.name)
        # 去除prompt
        X = X[:,:,-patch_num:,:]
        r=self.r_embed(status_token) 
        b=self.b_embed(status_token) 
        R=r.shape[-1]
        r_s=int(pow(R,1/2))
        modulation_scores = self.mlp(status_token)

        # input mask
        input_mask_radio = modulation_scores[:,:,:,0].unsqueeze(-1)
        input_mask_score  = modulation_scores[:,:,:,1:self.rank+1] 
        input_mask = torch.sigmoid((input_mask_score - input_mask_radio) / tau).unsqueeze(-1)

        # output mask
        output_mask_radio = modulation_scores[:,:,:,self.rank+1].unsqueeze(-1)
        output_mask_score = modulation_scores[:,:,:,self.rank+1+1: 2*self.rank+1+1]
        output_mask = torch.sigmoid((output_mask_score - output_mask_radio) / tau).unsqueeze(-2)
 
        r = r.expand(-1,self.n_vars,-1,-1)
        b = b.expand(-1,self.n_vars,-1,-1)
        r=rearrange(r,'batch_size n_vars patch_num (r1 r2) -> (batch_size n_vars patch_num) r1 r2',r1=r_s,r2=r_s)
        b=rearrange(b,'batch_size n_vars patch_num d -> (batch_size n_vars patch_num) d 1')
        
        lora_A=self.lora_A.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size,self.n_vars,patch_num,-1,-1) #拓展维度
        lora_B=self.lora_B.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size,self.n_vars,patch_num,-1,-1)
        lora_A = lora_A*input_mask.transpose(-1,-2)
        lora_B = lora_B*output_mask.transpose(-1,-2)
        lora_A = rearrange(lora_A,'batch_size n_vars patch_num r1 r2 -> (batch_size n_vars patch_num) r1 r2')
        lora_B = rearrange(lora_B,'batch_size n_vars patch_num r1 r2 -> (batch_size n_vars patch_num) r1 r2')

        W = torch.einsum('bij,bjk,bkl->bil', lora_A, r, lora_B)  # [1536, 16, 32]
        # (batch_size*self.n_vars*patch_num) input output
        W = W*b.transpose(-1, -2)
 
        X = rearrange(X, 'b n patch_num in_Feature -> (b n patch_num) 1 in_Feature', n=self.n_vars)   # [B*, 1, in_Feature]
        
        X = torch.einsum('bij,bjk->bik',X, W)
        # X [batch_size, n_vars, patch_num, out_features]
        X = rearrange(X, '(b n patch_num) 1 out_features  -> b n patch_num out_features', n=self.n_vars, patch_num=patch_num)
        
        part2 = self.scale * X

        if part1.size(2)>part2.size(2):
            part2 = torch.cat((torch.zeros([batch_size, self.n_vars, part1.size(2)-part2.size(2), part1.size(-1)],device=part1.device),part2), dim=2)
        output = part1 + part2

        if len(original_shape)==3:
            output = rearrange(output,'b n p d -> (b n) p d', n=self.n_continuous)
        return output
    

class LoRA(nn.Module):
    def __init__(self,
        pretrained_linear,
        wrapper_ref,
        name=None,
        n_continuous=0):

        super().__init__()
        self.pre_trained_weight = pretrained_linear
        self.in_features  = self.pre_trained_weight.in_features
        self.out_features  = self.pre_trained_weight.out_features
        # self.rank = wrapper_ref.config.rank

        self.rank = self.in_features // wrapper_ref.config.rank
        self.alpha = wrapper_ref.config.alpha

        self.name = name
        # 冻结，设置为不可训练
        self.pre_trained_weight.weight.requires_grad = False
        if self.pre_trained_weight.bias is not None:
            self.pre_trained_weight.bias.requires_grad = False
        # 低秩矩阵A和B
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank),requires_grad=True)
        # 对A矩阵进行高斯初始化
        nn.init.kaiming_normal_(self.lora_A, a = 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features),requires_grad=True)
        self.scale = self.alpha / self.rank


        
    def forward(self, X):
        part1 = self.pre_trained_weight(X)
        X = X @ self.lora_A @ self.lora_B
        part2 = self.scale * X
        output = part1 + part2
        return output