import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from mpu import EMT_Fusion
try:
    from utils import init_weights, get_padding
except:
    from .utils import init_weights, get_padding
import pdb

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.Audio["codebook_weight"], h.upsample_initial_channel, 7, 1, padding=3)) #承接的初始宽度是512维
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u,
                                # padding=(u//2 + u%2),
                                padding=(k - u )//2,
                                # output_padding=u%2
                )
            ))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    
    def forward(self, x): 
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(1, 32, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
                list(
                    reversed(
                        list(zip(h.upsample_rates, h.upsample_kernel_sizes))))):
            self.ups.append(
                weight_norm(
                    Conv1d(
                        32 * (2**i),
                        32 * (2**(i + 1)),
                        k,
                        u,
                        padding=((k - u) // 2)
                        # padding=(u//2 + u%2)
                    )))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 32 * (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(
                        list(reversed(h.resblock_kernel_sizes)),
                        list(reversed(h.resblock_dilation_sizes)))):
                self.resblocks.append(resblock(h, ch, k, d))
                self.normalize.append(
                    torch.nn.GroupNorm(ch // 16, ch, eps=1e-6, affine=True))
        self.conv_post = Conv1d(512, 512, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
        self.emt = EMT_Fusion(latent_dim = 512, input_dim = 512, 
                                   depth = 2, heads = 4, 
                                   dim_head = 512 // 4, 
                                   ff_expansion=4, attn_dropout=0., ff_dropout=0.)
    '''
    def fusion(self, x, sign_trait):
        # sign_trait = torch.randint(low=0, high=10, size=(32, 256)).to(x.device)
        sign_trait = sign_trait.unsqueeze(1)   # [32, 256] ——> [32, 1, 256] 
        sign_trait = sign_trait.expand(sign_trait.shape[0], x.shape[2], sign_trait.shape[2]) # [32, 1, 256] ——> [32, 50, 256]
        x = x.transpose(1,2) # [32, 512, 50] ——> [32, 50, 512]


        x = torch.cat((x, sign_trait), dim=2)  # [32, 50, 512] + [32, 50, 256] = [32, 50, 768]


        
        x = x.transpose(1,2) # [32, 50, 768] -> [32, 768, 50]
        
        return x
    '''
    def fusion(self, x, sign_trait):
        # sign_trait = torch.randint(low=0, high=10, size=(32, 256)).to(x.device)
        sign_trait = sign_trait.unsqueeze(1)   # [32, 256] ——> [32, 1, 256] 
        sign_trait = sign_trait.expand(sign_trait.shape[0], x.shape[2], sign_trait.shape[2]) # [32, 1, 256] ——> [32, 50, 256]
        x = x.transpose(1,2) # [32, 512, 50] ——> [32, 50, 512]
        
        x = self.emt(x, sign_trait)  # [32, 50, 512] and [32, 50, 256] to [32, 50, 512]
        
        x = x.transpose(1,2) # [32, 50, 512] -> [32, 512, 50]
        
        return x

    def forward(self, x, sign_trait):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x) # [32, 512, 50]

        x = self.fusion(x, sign_trait) #  [32, 512, 50] + [32, 256] = [32, 512, 50] fusion中没有一个线性层将768置换成512

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)  # [1024, 256]

    def forward(self, x): # [1600, 256]
        # compute Euclidean distance
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
            - 2 * torch.matmul(x, self.embedding.weight.T) # [1600, 1024]
        min_indicies = torch.argmin(d, 1)  # [1600]
        z_q = self.embedding(min_indicies) # [1600, 256]
        return z_q, min_indicies


class Quantizer(torch.nn.Module):
    def __init__(self, h, note):
        super(Quantizer, self).__init__()
        if note == 'Audio':
            self.h = h
            self.n_code_groups = self.h.Audio["n_code_groups"]
            self.n_codes = self.h.Audio["n_codes"]
            self.codebook_loss_lambda = self.h.Audio["codebook_loss_lambda"]
            self.commitment_loss_lambda = self.h.Audio["commitment_loss_lambda"]
            self.codebook_weight = self.h.Audio["codebook_weight"]
            self.residul_layer = int(self.h.Audio["residul_layer"])

        assert self.codebook_weight % self.n_code_groups == 0
        
        self.quantizer_module_residul_list = nn.ModuleList()
        for i in range(self.residul_layer):
            self.quantizer_module_residul_list.append(nn.ModuleList([
            Quantizer_module(self.n_codes, self.codebook_weight // self.n_code_groups) for _ in range(self.n_code_groups)
        ]))

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, self.codebook_weight)
        x = torch.split(x, self.codebook_weight // self.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []

        for _x, m in zip(x, self.quantizer_module_residul_list[idx]): # 元组x是2个，self.quantizer_modules也是两个(每个是Quantizer_module)
            _z_q, _min_indicies = m(_x) #_z_q:[235, 256]
            z_q.append(_z_q)
            min_indicies.append(_min_indicies) #B * T,
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
        loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
            + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
        z_q = xin + (z_q - xin).detach()
        z_q = z_q.transpose(1, 2)
        return z_q, loss, min_indicies

    def forward(self, xin):
        #B, C, T
        quantized_out = 0.0
        residual = xin
        all_losses = []
        all_indices = []
        for i in range(self.residul_layer):
            quantized, loss, indices = self.for_one_step(residual, i)  # 
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.extend(indices)  # 
            all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        return quantized_out, loss, all_indices

    def embed(self, x, need): # x 是forward返回的all_indices(序号记录)变成的tensor:[1, 235, 4] 原list长度4变成最后一个维度了,need是推理时需要的码本层级，我可以用4层训练，但是推理时只取2层
        #idx: N, T, 4
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2) # 拆分[1, 235, 4]的dim 2，拆成该维每个大小为1的，即变成list[[1, 235, 1], [1, 235, 1], [1, 235, 1], [1, 235, 1]]
        for i in range(min(self.residul_layer, need)):
            ret = []
            for j in range(self.n_code_groups):
                q = x[j] # q：[1, 235, 1] 某个记录表
                embed = self.quantizer_module_residul_list[i][j] # 对应码本
                q = embed.embedding(q.squeeze(-1)) # 查阅得到的[1, 235, 256]
                ret.append(q)
            ret = torch.cat(ret, -1) # 合并第一组两个codebook的查阅结果，得到[1, 235, h.codebook_weight]
            quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2) #N, C, T


