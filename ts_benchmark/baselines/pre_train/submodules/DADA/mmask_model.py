import torch
from torch import nn
import numpy as np
from einops import rearrange

from ts_benchmark.baselines.pre_train.submodules.DADA.layers.dilated_conv import DilatedConvEncoder
from ts_benchmark.baselines.pre_train.submodules.DADA.layers.adaptive_bottleneck import AdaptiveBottleNeck
from ts_benchmark.baselines.pre_train.submodules.DADA.layers.gradient_reverse import WarmStartGradientReverseLayer, GradientReverseFunction

import torch.nn.functional as F


class MMaskModel(nn.Module):
    def __init__(
            self,
            win_size=100,
            patch_len=5,
            mask_mode="symmetry",
            hidden_dim=64,
            repr_dim=256,
            depth=10,
            adp_bottleneck=True,
            bottleneck_dims=[16, 32, 64, 128, 192, 256],
            k=3,
            revin=False,
            backbone="dilated_conv",
            max_iters=1e5,
            # use_channel_processor=True,  # 控制是否使用通道处理器
            # channel_processor_dropout=0.1,  # 通道处理器的dropout率
            # channel_processor_mode = 'conv',  # 选择通道处理模式
            # channel_nums = 10,  # 通道数
            # use_multi_scale=True,
            # multi_patch_lens=[10, 20],
    ):
        super().__init__()
        # add
        self.win_size = win_size
        # RevIn
        self.revin = revin
        # Patch
        self.patch_len = patch_len
        if win_size % patch_len:
            self.patch_num = (win_size // patch_len) + 1
        else:
            self.patch_num = win_size // patch_len

        # Encoder
        self.mask_mode = mask_mode
        self.repr_dim = repr_dim
        self.input_embed = nn.Linear(patch_len, hidden_dim)
        self.encoder = Encoder(
            patch_len=patch_len,
            patch_num=self.patch_num,
            output_dims=repr_dim,
            hidden_dims=hidden_dim,
            depth=depth,
            backbone=backbone,
        )
        # BottleNeck
        self.adp_bottleneck = adp_bottleneck
        if adp_bottleneck:
            self.adaptive_bottleneck = AdaptiveBottleNeck(
                seq_len=self.patch_num,
                seq_dim=repr_dim,  # update
                repr_dim=repr_dim,
                bn_dims=bottleneck_dims,
                k=k,
            )
        else:
            assert len(bottleneck_dims) == 1
            self.bottleneck = nn.Sequential(
                nn.Linear(repr_dim, bottleneck_dims[0]),
                nn.GELU(),
                nn.Linear(bottleneck_dims[0], bottleneck_dims[0]),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dims[0], repr_dim),
            )
        # Decoder
        self.decoder = MLP_Decoder(repr_dim, patch_len)
        self.adv_decoder = MLP_Decoder(repr_dim, patch_len)
        self.grl = WarmStartGradientReverseLayer(hi=0.5, max_iters=max_iters, auto_step=True)


    def forward(self, x, grl=0):  # b x t x c
        # TEST OK
        B, T, dims = x.size()
        # print(f"Input shape: {x.shape}")
        # 0.normalization
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # 1.channel independence
        x = x.permute(0, 2, 1)  # b x c x t
        x = x.reshape(B * dims, T)  # b*c x t
        x_main = x
        # 2.do patch
        if T % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            padding = torch.zeros([B, (length - T)]).to(x.device)
            input = torch.cat([x, padding], dim=1)  # b*c x patch_num*patch_len
        else:
            length = T
            input = x  # b*c x t
        
        input_patch = input.unfold(dimension=-1, size=self.patch_len,
                                   step=self.patch_len)  # b*c x patch_num x patch_len
        input_patch = self.input_embed(input_patch)  # b*c x patch_num x hidden_dims
        # print(input_patch.shape)
        # 3.symmetry mask: generate copies
        if self.mask_mode == "symmetry":
            mask_1 = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims, self.patch_num))).to(
                torch.bool)  # b*c x patch_num
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0)  # b*c*2 x patch_num

            
            input_patch = input_patch.repeat(2, 1, 1)
            input_patch[mask] = 0  # patch symmetry mask
        else:
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims, self.patch_num))).to(
                torch.bool)  # b*c x patch_num
            input_patch[mask] = 0
        # 4.encoder
        repr = self.encoder(input_patch)  # b*c*2 x patch_num x repr_dim
        emb_c = repr.clone()
        # 5.adpBN
        balance_loss = torch.tensor(0., device=x.device, requires_grad=True)
        if self.adp_bottleneck:
            repr = torch.reshape(repr, (-1, self.patch_num * self.repr_dim))
            repr, balance_loss = self.adaptive_bottleneck(repr, repr, c=dims)
            repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))
        else:
            repr = self.bottleneck(repr)

        # 6.dual decoder
        if grl:
            gr_repr = self.grl(repr)
            out = self.adv_decoder(gr_repr)  # b*c*2 x patch_num*patch_len
        else:
            out = self.decoder(repr)  # b*c*2 x patch_num*patch_len

        # 7.symmetry mask: concat masked part
        if self.mask_mode == "symmetry":
            mask = mask.repeat(1, self.patch_len)  # b*c*2 x patch_num*patch_len
            out[~mask] = 0
            out = torch.reshape(out, (2, B * dims, self.patch_num * self.patch_len))  # 2 x b*c x patch_num*patch_len
            out = torch.sum(out, dim=0)  # b*c x patch_num*patch_len

        out = out[:, :T].reshape(B, dims, T)
        out = out.permute(0, 2, 1)  # b x t x c
        # de-Normalization
        if self.revin:
            out = out * stdev + means
        return out, emb_c, balance_loss

    def inference(self, x, mask_mode=None, copies=10, grl=0):
        if mask_mode is None:
            mask_mode = self.mask_mode
        B, T, dims = x.size()
        # 0.normalization
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # 1.channel independence
        x = x.permute(0, 2, 1)  # b x c x t
        x = x.reshape(B * dims, T)  # b*c x t
        x_main = x
        # 2.do patch
        if T % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            padding = torch.zeros([B, (length - T)]).to(x.device)
            input = torch.cat([x, padding], dim=1)  # b*c x patch_num*patch_len
        else:
            length = T
            input = x  # b*c x t
        input_patch = input.unfold(dimension=-1, size=self.patch_len,
                                   step=self.patch_len)  # b*c x patch_num x patch_len
        input_patch = self.input_embed(input_patch)  # b*c x patch_num x hidden_dims
        # 3.mask: generate copies
        if mask_mode == "symmetry":
            assert copies % 2 == 0, "The number of copies of symmetric mask must be an even number"
            mask_1 = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims * (copies // 2), self.patch_num))).to(
                torch.bool)
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0)  # b*c*copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0  # patch symmetry mask
        elif mask_mode == "random":
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims * copies, self.patch_num))).to(
                torch.bool)  # b*c*copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0
        elif mask_mode == "nomask":
            copies = 1
        # 4.encoder
        repr = self.encoder(input_patch)  # b*c*copies x patch_num x repr_dim
        emb_c = repr.clone()
        # 5.adpBN
        if self.adp_bottleneck:
            repr = torch.reshape(repr, (-1, self.patch_num * self.repr_dim))
            repr, balance_loss = self.adaptive_bottleneck(repr, repr)
            repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))

        if grl:
            gr_repr = self.grl(repr)
            out = self.adv_decoder(gr_repr)  # b*c*2 x patch_num*patch_len
        else:
            out = self.decoder(repr)
        # 6.norm decoder
        # out = self.decoder(repr)  # b*c*copies x patch_num*patch_len

        out = out.reshape(copies, B * dims, T)
        out = out[:, :, :T].reshape(copies, B, dims, T)
        out = out.permute(0, 1, 3, 2)  # copies x b x t x c
        # de-Normalization
        if self.revin:
            out = out * stdev.unsqueeze(dim=0).repeat(copies, 1, 1, 1) + means.unsqueeze(dim=0).repeat(copies, 1, 1, 1)
        return out, emb_c  # (copies) x b x t x c or (copies) x b x t

    def _create_scale_processor(self, patch_len, hidden_dim, repr_dim, depth):
        """创建独立的多尺度处理单元"""
        return nn.ModuleDict({
            'embed': nn.Linear(patch_len, hidden_dim),
            # 'encoder': Encoder(patch_len, self.win_size // patch_len, repr_dim, hidden_dim, depth),
            'mask_module': self._create_mask_processor()  # 共享掩码逻辑
        })

    def _create_mask_processor(self):
        """创建与原始模型一致的掩码处理器"""

        class MaskProcessor(nn.Module):
            def forward(self, x, mask_mode, B, dims, patch_num):
                # 对称掩码模式
                if mask_mode == "symmetry":
                    mask_1 = torch.from_numpy(
                        np.random.binomial(1, 0.5, size=(B * dims, patch_num))
                    ).to(torch.bool)
                    mask_2 = ~mask_1
                    mask = torch.cat([mask_1, mask_2], dim=0)
                    x = x.repeat(2, 1, 1)
                    x[mask] = 0
                # 非对称掩码模式
                else:
                    mask = torch.from_numpy(
                        np.random.binomial(1, 0.5, size=(B * dims, patch_num))
                    ).to(torch.bool)
                    x[mask] = 0
                return x

        return MaskProcessor()

    def _create_scale_processor_inference(self, patch_len, hidden_dim, repr_dim, depth):
        """创建独立的多尺度处理单元"""
        return nn.ModuleDict({
            'embed': nn.Linear(patch_len, hidden_dim),
            # 'encoder': Encoder(patch_len, self.win_size // patch_len, repr_dim, hidden_dim, depth),
            'mask_module': self._create_mask_processor_inference()  # 共享掩码逻辑
        })

    def _create_mask_processor_inference(self):
        """创建与原始模型一致的掩码处理器"""

        class MaskProcessor_inference(nn.Module):
            def forward(self, x, mask_mode, B, dims, patch_num, copies):
                if mask_mode == "symmetry":
                    assert copies % 2 == 0, "The number of copies of symmetric mask must be an even number"
                    mask_1 = torch.from_numpy(
                        np.random.binomial(1, 0.5, size=(B * dims * (copies // 2), patch_num))).to(torch.bool)
                    mask_2 = ~mask_1
                    mask = torch.cat([mask_1, mask_2], dim=0)  # b*c*copies x patch_num
                    x = x.repeat(copies, 1, 1)
                    x[mask] = 0  # patch symmetry mask
                elif mask_mode == "random":
                    mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims * copies, patch_num))).to(
                        torch.bool)  # b*c*copies x patch_num
                    x = x.repeat(copies, 1, 1)
                    x[mask] = 0
                elif mask_mode == "nomask":
                    copies = 1
                return x

        return MaskProcessor_inference()

    def _create_patches(self, x, patch_len):
        """通用patch创建方法"""
        B, T = x.shape
        if T % patch_len != 0:
            pad_len = (T // patch_len + 1) * patch_len - T
            x = F.pad(x, (0, pad_len))
        return x.unfold(1, patch_len, patch_len)  # [B, patch_num, pl]


class Encoder(nn.Module):
    def __init__(self, patch_len, patch_num, output_dims=512, hidden_dims=64, depth=10, backbone="dilated_conv"):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.backbone = backbone
        if backbone == "dilated_conv":
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
            )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x hidden_dims x patch_num
        repr = self.repr_dropout(self.feature_extractor(x))  # B x output_dims x patch_num
        repr = repr.permute(0, 2, 1)
        return repr  # B x patch_num x output_dims


class MLP_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        hidden_dim = int(input_dims * 2)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dims),
        )

        # lora
        # self.net = nn.Sequential(
        #     nn.GELU(),
        #     lora.Linear(input_dims, hidden_dim),
        #     nn.GELU(),
        #     lora.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     lora.Linear(hidden_dim, output_dims),
        # )

        self.flatten = nn.Flatten(-2)

    def forward(self, x):  # b*c x patch_num x repr_dim
        x = self.net(x)  # b*c x patch_num x patch_len
        x = self.flatten(x)  # b*c x patch_num*patch_len
        return x


class Flatten_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(input_dims, output_dims),
        )

    def forward(self, x):  # b*c x patch_num x repr_dim
        x = self.net(x)  # b*c x patch_num*patch_len
        return x


class MLP_Classifier(nn.Module):
    def __init__(self, input_dims, abnorm_class, patch_len):
        super().__init__()
        hidden_dim = int(input_dims * 2)
        self.patch_len = patch_len
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, abnorm_class),
            nn.Sigmoid(),
        )

    def forward(self, x):  # b x c x patch_num x repr_dim
        # 1. 平均掉通道维：合并多通道信息
        x = x.mean(dim=1)  # [batch_size, patch_num, repr_dim]
        # 2. 将 patch_num 和 repr_dim 映射到时间点
        x = self.net(x)  # [batch_size, patch_num, 1]
        # 3. 重复 patch_len 次以匹配原始序列长度
        x = x.repeat_interleave(self.patch_len, dim=1)  # [batch_size, patch_num * patch_len, 1]
        return x.squeeze(-1)  # 去掉最后一维，输出形状: [batch_size, seq_len]


class ChannelProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1, mode='conv'):
        """
        增强版通道处理模块
        Args:
            input_dim: 输入通道维度
            hidden_dim: 隐藏层维度，默认为input_dim的2倍
            dropout: dropout率
            mode: 处理模式，可选 'conv'（卷积）、'attention'（自注意力）或 'mlp'（多层感知机）
        """
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 2
        self.mode = mode

        if mode == 'conv':
            # 深度可分离卷积+通道混合
            self.depthwise = nn.Conv1d(
                input_dim, input_dim, kernel_size=3,
                padding=1, groups=input_dim)
            self.pointwise = nn.Conv1d(
                input_dim, hidden_dim, kernel_size=1)
            self.projection = nn.Conv1d(
                hidden_dim, input_dim, kernel_size=1)
            self.norm = nn.LayerNorm(input_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
        elif mode == 'attention':
            # 多头自注意力处理通道维度
            pass
        elif mode == 'mlp':
            # MLP处理通道维度
            pass

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
        Returns:
            处理后的张量，形状与输入相同
        """
        residual = x
        if self.mode == 'conv':
            # 转换维度处理
            x = x.permute(0, 2, 1)  # [batch, channels, seq]
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.projection(x)
            x = x.permute(0, 2, 1)  # 恢复维度
            x = self.norm(x + residual)
            return x
        elif self.mode == 'attention':
            # 自注意力处理
            return x
        else:
            # mlp
            return x
