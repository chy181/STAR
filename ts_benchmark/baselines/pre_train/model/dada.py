import torch
from torch import nn
import sys
sys.path.insert(0, 'ts_benchmark/baselines/pre_train/submodules/DADA')
from einops import rearrange
from ts_benchmark.baselines.pre_train.submodules.DADA.mmask_model import MMaskModel

class DadaModel(nn.Module):
    def __init__(
        self,
        config,
        **kwargs
    ):
        super().__init__()

        self.patch_len = 5 
        self.config = config
        self.phase = 'train'
        self.model = MMaskModel(win_size=config.seq_len,
            patch_len=self.patch_len,
            mask_mode="symmetry",
            hidden_dim=64,
            repr_dim=256,
            depth=10,
            adp_bottleneck=True,
            bottleneck_dims=[16, 32, 64, 128, 192, 256],
            k=3,
            revin=False,
            backbone="dilated_conv",
            max_iters=100000.0) 
        
        self.model.load_state_dict(torch.load("ts_benchmark/baselines/pre_train/checkpoints/dada/dada.pth"))
     
    def forward(self, inputs):
        
        outputs, emb_c, _ = self.model(inputs)
        # b, t, c = inputs.shape
        # emb_c = rearrange(emb_c, '(b c copies) p d -> b c p d copies', b=b, c=c, copies=2)
        # emb_c = emb_c.mean(-1)

        b, t, c = inputs.shape
        if self.phase == 'test':
            outputs, emb_c = self.model.inference(inputs,mask_mode='symmetry',copies=10)
            outputs = torch.mean(outputs, dim=0)
            
            emb_c = rearrange(emb_c, '(b c copies) p d -> b c p d copies', b=b, c=c, copies=10)
            emb_c = emb_c.mean(-1)

        else:
            outputs, emb_c, _ = self.model(inputs)
            emb_c = rearrange(emb_c, '(b c copies) p d -> b c p d copies', b=b, c=c, copies=2)
            emb_c = emb_c.mean(-1)
        return outputs, {'emb_c':emb_c}
