import torch
import torch.nn as nn
from .SPE import SPE
from .blocks import SelfAttention1d, CrossAttention1d
from einops import rearrange, reduce, repeat
import gin
from .Encoders import Encoder1D


@gin.configurable
class ConvBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size,
                 n_stems=1,
                 act=nn.SiLU,
                 res=True):
        super().__init__()
        self.res = res
        self.conv1 = nn.Conv1d(in_c + time_cond_channels,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")
        self.gn1 = nn.GroupNorm(min(16, out_c), out_c)
        self.conv2 = nn.Conv1d(out_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")
        self.gn2 = nn.GroupNorm(min(16, out_c), out_c)
        self.act = act()

        self.time_mlp = nn.Sequential(nn.Linear(time_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))
        self.n_stems = n_stems
        if n_stems == 1:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))
        else:
            self.cond_mlp = nn.ModuleList()
            for _ in range(n_stems):
                self.cond_mlp.append(nn.Sequential(nn.Linear(cond_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c)))

        if in_c != out_c:
            self.to_out = nn.Conv1d(in_c // 2,
                                    out_c,
                                    kernel_size=1,
                                    padding="same")
        else:
            self.to_out = nn.Identity()

    def forward(self, x, time=None, skip=None, zsem=None, time_cond=None):
        if self.res:
            res = x.clone()

        if skip is not None:
            x = torch.cat([x, skip], axis=1)

        if time_cond is not None:
            if self.n_stems == 1:
                x = torch.cat([x, time_cond], axis=1)
            else:
                x = torch.cat([x, torch.stack(time_cond).max(axis=0).values], axis=1)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)

        time = self.time_mlp(time)
        time_mult, time_add = torch.split(time, time.shape[-1] // 2, -1)
        x = x * time_mult[:, :, None] + time_add[:, :, None]

        if self.n_stems == 1:
            zsem = self.cond_mlp(zsem)
            zsem_mult, zsem_add = torch.split(zsem, zsem.shape[-1] // 2, -1)
            x = x * zsem_mult[:, :, None] + zsem_add[:, :, None]
        else:
            x_list = []
            for i in range(self.n_stems):
                zsem_ = self.cond_mlp[i](zsem[i])
                zsem_mult, zsem_add = torch.split(zsem_, zsem_.shape[-1] // 2, -1)
                x_list.append(x * zsem_mult[:, :, None] + zsem_add[:, :, None])
            x = torch.stack(x_list).mean(axis=0)

        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)

        x = self.act(x)

        if self.res:
            return x + self.to_out(res)

        return x

    def load_weights(self, state_dict):
        conv1_weights = {}
        gn1_weights = {}
        conv2_weights = {}
        gn2_weights = {}
        time_mlp_weights = {}
        cond_mlp_weights = {}
        to_out_weights = {}
        for key, value in state_dict.items():
            if key.startswith('conv1.'):
                new_key = key.replace('conv1.', '', 1)
                conv1_weights[new_key] = value
            elif key.startswith('gn1.'):
                new_key = key.replace('gn1.', '', 1)
                gn1_weights[new_key] = value
            elif key.startswith('conv2.'):
                new_key = key.replace('conv2.', '', 1)
                conv2_weights[new_key] = value
            elif key.startswith('gn2.'):
                new_key = key.replace('gn2.', '', 1)
                gn2_weights[new_key] = value
            elif key.startswith('time_mlp.'):
                new_key = key.replace('time_mlp.', '', 1)
                time_mlp_weights[new_key] = value
            elif key.startswith('cond_mlp.'):
                new_key = key.replace('cond_mlp.', '', 1)
                cond_mlp_weights[new_key] = value
            elif key.startswith('to_out.'):
                new_key = key.replace('to_out.', '', 1)
                to_out_weights[new_key] = value
        self.conv1.load_state_dict(conv1_weights)
        self.gn1.load_state_dict(gn1_weights)
        self.conv2.load_state_dict(conv2_weights)
        self.gn2.load_state_dict(gn2_weights)
        self.time_mlp.load_state_dict(time_mlp_weights)
        self.to_out.load_state_dict(to_out_weights)
        if self.n_stems == 1:
            self.cond_mlp.load_state_dict(cond_mlp_weights)
        else:
            for i in range(self.n_stems):
                self.cond_mlp[i].load_state_dict(cond_mlp_weights)


@gin.configurable
class EncoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size=3,
                 ratio=2,
                 n_stems=1,
                 act=nn.SiLU,
                 use_self_attn=False):
        super().__init__()
        self.conv = ConvBlock1D(in_c=in_c,
                                out_c=in_c,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                n_stems=n_stems,
                                act=act)

        self.self_attn = SelfAttention1d(
            in_c, 4) if use_self_attn else nn.Identity()

        if ratio == 1:
            self.pool = nn.Conv1d(in_c,
                                  out_c,
                                  kernel_size=kernel_size,
                                  padding="same")
        else:
            self.pool = nn.Conv1d(in_c,
                                  out_c,
                                  kernel_size=kernel_size,
                                  stride=ratio,
                                  padding=(kernel_size) // 2)

    def forward(self,
                inputs,
                time,
                cond=None,
                time_cond=None,
                zsem=None,
                context=None):
        skip = self.conv(inputs, time=time, zsem=zsem, time_cond=time_cond)
        skip = self.self_attn(skip)
        x = self.pool(skip)
        return x, skip

    def load_weights(self, state_dict):
        conv_weights = {}
        attn_weights = {}
        pool_weights = {}
        for key, value in state_dict.items():
            if key.startswith('conv.'):
                new_key = key.replace('conv.', '', 1)
                conv_weights[new_key] = value
            elif key.startswith('self_attn.'):
                new_key = key.replace('self_attn.', '', 1)
                attn_weights[new_key] = value
            elif key.startswith('pool.'):
                new_key = key.replace('pool.', '', 1)
                pool_weights[new_key] = value
        self.conv.load_weights(conv_weights)
        self.self_attn.load_state_dict(attn_weights)
        self.pool.load_state_dict(pool_weights)


@gin.configurable
class MiddleBlock1D(nn.Module):

    def __init__(
        self,
        in_c,
        time_cond_channels,
        time_channels,
        cond_channels,
        kernel_size=3,
        ratio=2,
        n_stems=1,
        act=nn.SiLU,
        use_self_attn=False,
    ):
        super().__init__()
        self.conv = ConvBlock1D(in_c=in_c,
                                out_c=in_c,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                n_stems=n_stems,
                                act=act)
        self.self_attn = SelfAttention1d(
            in_c, in_c // 32) if use_self_attn else nn.Identity()

    def forward(self, x, time, time_cond=None, zsem=None):
        x = self.conv(x, time=time, zsem=zsem, time_cond=time_cond)
        x = self.self_attn(x)
        return x

    def load_weights(self, state_dict):
        conv_weights = {}
        attn_weights = {}
        for key, value in state_dict.items():
            if key.startswith('conv.'):
                new_key = key.replace('conv.', '', 1)
                conv_weights[new_key] = value
            elif key.startswith('self_attn.'):
                new_key = key.replace('self_attn.', '', 1)
                attn_weights[new_key] = value
        self.conv.load_weights(conv_weights)
        self.self_attn.load_state_dict(attn_weights)


@gin.configurable
class DecoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels,
                 kernel_size,
                 act=nn.SiLU,
                 ratio=2,
                 n_stems=1,
                 use_self_attn=False,
                 skip_size=None):
        super().__init__()
        if ratio == 1:
            if in_c == out_c:
                self.up = nn.Identity()
            else:
                self.up = nn.Conv1d(in_c,
                                    out_c,
                                    kernel_size=3,
                                    stride=1,
                                    padding="same")
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=ratio),
                nn.Conv1d(in_c, out_c, kernel_size=3, stride=1,
                          padding="same"))

        self.conv = ConvBlock1D(in_c=out_c + out_c,
                                out_c=out_c,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels=cond_channels,
                                kernel_size=kernel_size,
                                n_stems=n_stems,
                                act=act)
        self.self_attn = SelfAttention1d(
            out_c, 4) if use_self_attn else nn.Identity()

    def forward(self, x, skip, time, time_cond=None, zsem=None, context=None):
        x = self.up(x)
        x = self.conv(x, time=time, skip=skip, zsem=zsem, time_cond=time_cond)
        x = self.self_attn(x)
        return x

    def load_weights(self, state_dict):
        conv_weights = {}
        up_weights = {}
        for key, value in state_dict.items():
            if key.startswith('conv.'):
                new_key = key.replace('conv.', '', 1)
                conv_weights[new_key] = value
            elif key.startswith('up.'):
                new_key = key.replace('up.', '', 1)
                up_weights[new_key] = value
        self.conv.load_weights(conv_weights)
        self.up.load_state_dict(up_weights)


@gin.configurable
class UNET1D(nn.Module):

    def __init__(self,
                 in_size=128,
                 channels=[128, 128, 256, 256],
                 ratios=[2, 2, 2, 2, 2],
                 kernel_size=5,
                 cond={},
                 time_channels=64,
                 time_cond_in_channels=1,
                 time_cond_channels=64,
                 z_channels=32,
                 n_attn_layers=0,
                 n_stems=1):

        super().__init__()
        self.channels = channels
        self.time_cond_channels = time_cond_channels
        self.n_stems = n_stems

        n = len(self.channels)

        if time_channels == 0:
            self.time_emb = lambda _: torch.empty(0)
        else:
            self.time_emb = SPE(time_channels)

        self.cond_modules = nn.ModuleDict()
        self.cond_keys = list(cond.keys())

        c_channels = 0
        for key, p in cond.items():
            if p["num_classes"] == 0:
                self.cond_modules[key] = nn.Sequential(
                    nn.Linear(1, p["emb_dim"]), nn.ReLU())

            else:
                self.cond_modules[key] = nn.Embedding(p["num_classes"] + 1,
                                                      p["emb_dim"])
            c_channels += p["emb_dim"]

        cond_channels = c_channels + z_channels

        if time_cond_channels:
            if n_stems == 1:
                self.cond_emb_time = nn.ModuleList()
                self.cond_emb_time.append(
                    nn.Sequential(
                        nn.Conv1d(time_cond_in_channels,
                                    time_cond_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding="same"), nn.SiLU()))
                for i in range(0, n):
                    self.cond_emb_time.append(
                        nn.Sequential(
                            nn.Conv1d(time_cond_channels,
                                        time_cond_channels,
                                        kernel_size=kernel_size,
                                        stride=ratios[i],
                                        padding="same" if ratios[i] == 1 else
                                        (kernel_size) // 2), nn.SiLU()))
            else:
                self.cond_emb_time = nn.ModuleList()
                for _ in range(n_stems):
                    cond_emb_time = nn.ModuleList()
                    cond_emb_time.append(
                        nn.Sequential(
                            nn.Conv1d(time_cond_in_channels,
                                        time_cond_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding="same"), nn.SiLU()))
                    for i in range(0, n):
                        cond_emb_time.append(
                            nn.Sequential(
                                nn.Conv1d(time_cond_channels,
                                            time_cond_channels,
                                            kernel_size=kernel_size,
                                            stride=ratios[i],
                                            padding="same" if ratios[i] == 1 else
                                            (kernel_size) // 2), nn.SiLU()))
                    self.cond_emb_time.append(cond_emb_time)


        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()

        self.down_layers.append(
            EncoderBlock1D(in_c=in_size,
                           out_c=channels[0],
                           time_channels=time_channels,
                           time_cond_channels=time_cond_channels,
                           cond_channels=cond_channels,
                           kernel_size=kernel_size,
                           ratio=ratios[0], 
                           n_stems=n_stems))
        comp_ratios = []
        cur_ratio = 1
        for r in ratios:
            cur_ratio *= r
            comp_ratios.append(cur_ratio)

        for i in range(1, n):
            self.down_layers.append(
                EncoderBlock1D(in_c=channels[i - 1],
                               out_c=channels[i],
                               time_channels=time_channels,
                               time_cond_channels=time_cond_channels,
                               cond_channels=cond_channels,
                               kernel_size=kernel_size,
                               ratio=ratios[i],
                               n_stems=n_stems,
                               use_self_attn=i >= n - n_attn_layers))
            self.up_layers.append(
                DecoderBlock1D(in_c=channels[n - i],
                               out_c=channels[n - i - 1],
                               time_channels=time_channels,
                               time_cond_channels=time_cond_channels,
                               cond_channels=cond_channels,
                               kernel_size=kernel_size,
                               ratio=ratios[n - i],
                               n_stems=n_stems,
                               use_self_attn=i <= (n_attn_layers)))

        self.up_layers.append(
            DecoderBlock1D(in_c=channels[0],
                           out_c=in_size,
                           time_channels=time_channels,
                           time_cond_channels=time_cond_channels,
                           cond_channels=cond_channels,
                           kernel_size=kernel_size,
                           ratio=ratios[0],
                           n_stems=n_stems))

        self.middle_block = MiddleBlock1D(
            in_c=channels[-1],
            time_channels=time_channels,
            cond_channels=cond_channels,
            time_cond_channels=time_cond_channels,
            kernel_size=kernel_size,
            n_stems=n_stems,
            use_self_attn=n_attn_layers > 0)

    def forward(self, x, time=None, cond=None, time_cond=None, zsem=None):
        time = self.time_emb(time).to(x)
        self.cond_modules.cpu()

        cond_embs = []
        for key in self.cond_keys:
            on = True
            if key == "key":
                val = self.cond_modules[key](cond[key] + 1)[:, 0]
            else:
                val = self.cond_modules[key](cond[key])

            cond_embs.append(val)

        if len(cond_embs) > 0:
            cond_embs = torch.cat(cond_embs, axis=-1).to(x)
        else:
            cond_embs = torch.empty(0).to(x)
        if self.n_stems > 1:
            cond_embs = [cond_embs.clone() for _ in range(self.n_stems)]

        skips = []
        time_conds = []

        if self.time_cond_channels:
            for i, layer in enumerate(self.down_layers):

                if time_cond is not None:
                    if self.n_stems == 1:
                        time_cond = self.cond_emb_time[i](time_cond)
                    else:
                        time_cond = [self.cond_emb_time[j][i](time_cond[j]) for j in range(self.n_stems)]

                x, skip = layer(x, time=time, time_cond=time_cond, zsem=zsem)

                time_conds.append(time_cond)
                skips.append(skip)

            if self.n_stems == 1:
                time_cond = self.cond_emb_time[-1](time_cond)
            else:
                time_cond = [self.cond_emb_time[j][-1](time_cond[j]) for j in range(self.n_stems)]

            x = self.middle_block(x, time=time, time_cond=time_cond, zsem=zsem)

            for layer in self.up_layers:
                skip = skips.pop(-1)
                time_cond = time_conds.pop(-1)

                x = layer(x,
                          skip=skip,
                          time=time,
                          time_cond=time_cond,
                          zsem=zsem)

            return x

        else:
            for layer in self.down_layers:
                x, skip = layer(x, time, cond_embs, zsem=zsem)
                skips.append(skip)

            x = self.middle_block(x, time, cond_embs, zsem=zsem)

            for layer in self.up_layers:
                skip = skips.pop(-1)
                x = layer(x, skip, time, cond_embs, zsem=zsem)
            return x
        
    def load_weights(self, state_dict):
        time_emb_weights = {}
        cond_emb_time_weights = {}
        up_layers_weights = {i: {} for i in range(len(self.up_layers))}
        down_layers_weights = {i: {} for i in range(len(self.down_layers))}
        middle_block_weights = {}
        for key, value in state_dict.items():
            if key.startswith('time_emb.'):
                new_key = key.replace('time_emb.', '', 1)
                time_emb_weights[new_key] = value
            elif key.startswith('cond_emb_time.'):
                new_key = key.replace('cond_emb_time.', '', 1)
                cond_emb_time_weights[new_key] = value
            elif key.startswith('up_layers.'):
                new_key = key.replace('up_layers.', '', 1)
                up_layers_weights[int(new_key[0])][new_key[2:]] = value  
            elif key.startswith('down_layers.'):
                new_key = key.replace('down_layers.', '', 1)
                down_layers_weights[int(new_key[0])][new_key[2:]] = value
            elif key.startswith('middle_block.'):
                new_key = key.replace('middle_block.', '', 1)
                middle_block_weights[new_key] = value
        self.time_emb.load_state_dict(time_emb_weights)  
        for i, weights in up_layers_weights.items():
            self.up_layers[i].load_weights(weights)
        for i, weights in down_layers_weights.items():
            self.down_layers[i].load_weights(weights)
        self.middle_block.load_weights(middle_block_weights)
        if self.n_stems == 1:
            self.cond_emb_time.load_state_dict(cond_emb_time_weights)
        else:
            for i in range(self.n_stems):
                self.cond_emb_time[i].load_state_dict(cond_emb_time_weights)

