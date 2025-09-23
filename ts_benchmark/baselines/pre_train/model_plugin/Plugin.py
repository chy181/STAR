from torch import nn
from einops import rearrange
from ts_benchmark.baselines.pre_train.model_plugin.layers.embedding import  StatusEmbedding
from ts_benchmark.baselines.pre_train.model_plugin.layers.star import  STAR
from ts_benchmark.baselines.pre_train.model_plugin.layers.LoRA import LinearLoRA,LoRA

class Plugin(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        self.mode = config.mode
        self.backbone = backbone
        self.config = config
        if self.mode == 'LoRA':
            self.config = config
            self._freeze()
            self._replace_layers_LoRA(self.backbone)

        if self.mode == 'plugin':

            if config.backbone == 'DADA':
                # Dada
                stride = 5
                patch_len = 5
                backbone_dim = 256
            elif config.backbone == 'UniTS':
                # UniTS
                stride = backbone.args.stride
                patch_len = backbone.args.patch_len
                backbone_dim = 32

            elif config.backbone == 'Moment':
                stride = backbone.model.config.patch_len
                patch_len = backbone.model.config.patch_stride_len
                backbone_dim = 512

            elif config.backbone == 'Timer':
                stride = 96
                patch_len = 96
                backbone_dim = 512

            self.n_discrete  = config.n_discrete
            self.n_continuous = config.n_continuous

            self.status_embedding = StatusEmbedding(num_shared_experts=config.num_shared_experts, num_experts=config.num_experts, seq_len=config.seq_len, enc_in=config.discrete_nums, K=config.K ,d_model=config.d_model,patch_len=patch_len, stride=stride)
            
            self.star_c = STAR(backbone_dim, config.d_model)
            self.star_d = nn.Sequential(nn.Linear(config.d_model,config.d_model),nn.ReLU(),nn.Linear(config.d_model,config.d_model))
            self.status_token = None
            
            self.config = config
            self.stride = stride
            self.patch_len = patch_len
            self._freeze()
            self._replace_layers(self.backbone)


    def forward(self, continuous, discrete):
        if self.config.mode=='backbone' or self.config.mode=='LoRA':
            rec_c, _ = self.backbone(continuous)
            return {"rec_c":rec_c}
        
        elif self.mode=='plugin':
            continuous0 = continuous

            # Discrete embedding
            discrete, self.discrete_token, L_importance, weight0 = self.status_embedding(discrete)
            rec_c, result_dict = self.backbone(continuous0)

            emb_c = result_dict['emb_c'][:,:,-self.discrete_token.size(2):,:]
            
            emb_d = self.discrete_token.squeeze(1).unsqueeze(2)
            emb_c = self.star_c(emb_c).squeeze(1).unsqueeze(2)

            return {"L_importance":L_importance, "emb_c":emb_c, "emb_d":emb_d, "rec_c":rec_c,"discrete":discrete, "weight0":weight0}
    
    def _replace_layers_LoRA(self, module, father=''):
        for name, child_module in module.named_children():

            if isinstance(child_module, nn.Linear):
                # UniTS
                module_list = ['value_embedding', 'qkv']
                for replaced_partten in module_list:
                    if replaced_partten in father or replaced_partten in name:
                        setattr(module, name, LoRA(child_module, wrapper_ref=self, name=father+'.'+name ))
            else:
                self._replace_layers_LoRA(child_module, father= father+'.'+name ) # 递归进入子模块
        pass
    
    def _replace_layers(self, module, father=''):
        for name, child_module in module.named_children():

            if isinstance(child_module, nn.Linear):
                # UniTS
                module_list = ['value_embedding', 'qkv']
                for replaced_partten in module_list:
                    if replaced_partten in father or replaced_partten in name:
                        setattr(module, name, LinearLoRA(child_module, wrapper_ref=self, name=father+'.'+name, n_continuous = self.n_continuous ))

            else:
                self._replace_layers(child_module, father= father+'.'+name ) # 递归进入子模块
        pass

    def _freeze(self, ):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    def _unfreeze(self, ):
         for name, param in self.backbone.named_parameters():
             param.requires_grad = True

