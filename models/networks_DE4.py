
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from .transformer_block import Block, get_sinusoid_encoding, Unfold3D, Upfold3D
from unfoldNd import UnfoldNd
from torch.autograd import Variable
from .token_performer import Token_performer
from axial_attention import AxialAttention, AxialPositionalEmbedding


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])



def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'instance3D':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def weights_init3D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net



def define_G(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = GAN_local_trans()

    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, n_layers_D, norm='instance3D', use_sigmoid=False, num_D=1, getIntermFeat=False,
             gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator3D(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init3D)
    return netD

def discriminate(D, fake_pool, input_label, test_image, use_pool=False):
    input_concat = torch.cat((input_label, test_image.detach()), dim=1)
    if use_pool:
        fake_query = fake_pool.query(input_concat)
        return D.forward(fake_query)
    else:
        return D.forward(input_concat)


def define_S(input_nc, output_nc, ngf, netS, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    net = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    elif netS == 'aCFS-unet':
        net = UnetGenerator_SEG(input_nc, output_nc, ngf, trilinear=True, use_duse=True).cuda()   # UNet with a-CFS
    else:
        raise NotImplementedError('Segmentor model name [%s] is not recognized' % netS)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def feature_loss(ori_img, syn_img, pred_real, pred_fake, ext_discriminator):
    criterionFeat = torch.nn.L1Loss()
    loss_G_GAN_Feat = 0
    D_weights = 1.0 / 3
    for i in range(3):
        for j in range(len(pred_fake[i]) - 1):
            loss_G_GAN_Feat += D_weights * \
                                criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
    ori_img = ori_img.expand(-1, 3, -1, -1, -1)
    syn_img = syn_img.expand(-1, 3, -1, -1, -1)
    feat_res_real = ext_discriminator(ori_img)
    feat_res_fake = ext_discriminator(syn_img)
    loss_G_GAN_Feat_ext = 0
    res_weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    feature_level = ['layer1', 'layer2', 'layer3', 'layer4']
    for tmp_i in range(len(feature_level)):
        loss_G_GAN_Feat_ext += criterionFeat(feat_res_real[feature_level[tmp_i]].detach(),
                                            feat_res_fake[feature_level[tmp_i]]) * res_weights[tmp_i]

    return loss_G_GAN_Feat, loss_G_GAN_Feat_ext

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


##################################################################################
# Normalization layers
##################################################################################

class Trans_global(nn.Module):


    def __init__(self, embed_dim=64, depth=9, num_heads=2, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        self.sw = UnfoldNd(kernel_size=3, stride=1, padding=1)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=16 ** 3, d_hid=embed_dim), requires_grad=False)


        self.bot_proj = nn.Linear(3 ** 3, embed_dim)
        self.bot_proj2 = nn.Linear(embed_dim, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.to(torch.float32)
        # print('$$$$$$')
        # print(x.size())
        # print(self.PE.size())
        x = self.sw(x).transpose(1, 2)
        # print(x.size())
        x = self.bot_proj(x)
        # print(x.size())
        x = x + self.PE
        for blk in self.bottleneck:
            x = blk(x)
        # print(x.size())
        x = self.bot_proj2(x)
        # print(x.size())
        x = x.transpose(1, 2)
        # print(x.size())
        B, C, HW = x.shape
        x = x.reshape(B, C, 16, 16, 16)
        # print(x.size())
        return x


class GAN_local_trans(nn.Module):
    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 64, 128, 256, 512],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):  # img_size = opt.patch_size
        super().__init__()
        self.GlobalGenerator = Trans_global()
        self.down_blocks = nn.ModuleList(
            [Unfold3D(in_channel=1, out_channel=64, patch=3, stride=1, padding=1),  # 64
             Unfold3D(in_channel=64, out_channel=128, patch=3, stride=2, padding=1),  # 32
             Unfold3D(in_channel=128, out_channel=192, patch=3, stride=2, padding=1),  # 16
             Unfold3D(in_channel=256, out_channel=512, patch=3, stride=2, padding=1)])  # 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2)) for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [Upfold3D(in_channel=channels[-(i + 1)],
                          out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                          up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                          padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(Upfold3D(in_channel=channels[1],
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[1] / down_ratio[0]),
                                           padding=int((patch[0] - 1) / 2)))
        else:

            self.up_blocks = (nn.ModuleList(
                [Upfold3D(in_channel=2 * channels[-(i + 1)],
                          out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                          up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                          padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(Upfold3D(in_channel=2 * channels[1],
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[2] / down_ratio[1]),
                                           padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(Upfold3D(in_channel=channels[1] + 1,
                                           out_channel=channels[1], patch=patch[0],
                                           up_scale=int(down_ratio[1] / down_ratio[0]),
                                           padding=int((patch[0] - 1) / 2)))
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1] * patch[-1], embed_dim)
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])

        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        Global_feat = self.GlobalGenerator(F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=True))
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)

            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            a = []
            # print('! syn down !')
            for i, down in enumerate(self.down_blocks[:-1]):
                x = x.to(torch.float32)
                x = down(x)
                B, HW, C = x.shape

                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))

                if i == 2:
                    # print(x.size(), Global_feat.size())

                    x = torch.cat((x, Global_feat), dim=1)

                SC.append(x)

                # print(x.size())
            x = self.down_blocks[-1](x, attention=False)
            # print('%%%%%%')
            # print(x.size())
            x = self.bot_proj(x)
            # print(x.size())
            x = x + self.PE
            # print('@@@@@')
            # print(x.size(), self.PE.size())
            # print(x.size())
            for blk in self.bottleneck:
                x = blk(x)
            # print(x.size())
            x = self.bot_proj2(x).transpose(1, 2)
            # print(x.size())
            B, C, HW = x.shape

            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            # print('!!!')
            # print(x.size())
            for i, up in enumerate(self.up_blocks[:-2]):
                x = up(x, SC=SC[-(i + 1)], reshape=True, size=[x.shape[2], x.shape[3], x.shape[4]])
            for up in self.up_blocks[-2:-1]:
                x = up(x, SC=SC[0], reshape=True,
                       size=[x.shape[2], x.shape[3], x.shape[4]])

            x = self.up_blocks[-1](x, SC=x0, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        # print(SC[0].size(), SC[1].size(), SC[2].size(), SC[3].size())
        list = []
        list.append(self.tanh(x))
        list.append(SC)
        # print(len(list), len(list[1]))
        # return self.tanh(x)
        return list


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(input)

class MultiscaleDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator3D, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator3D(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        input = input.float()
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)



class UnetGenerator_SEG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, trilinear=True, use_duse=False):
        super(UnetGenerator_SEG, self).__init__()

        self.n_channels = input_nc
        self.n_classes = output_nc
        self.trilinear = trilinear
        self.inc = DoubleConv(self.n_channels, ngf, use_duse=False)
        self.down1 = Down(ngf, 2*ngf, use_duse=False)
        self.down2 = Down(2*ngf, 4*ngf, use_duse=False)
        self.down3 = Down(4*ngf, 8*ngf, use_duse=False)
        self.down4 = Down(8*ngf, 16*ngf, use_duse=False)

        # self.up1 = Up(16*ngf, 8*ngf, use_duse=use_duse)
        # self.up2 = Up(8*ngf, 4*ngf, use_duse=use_duse)
        # self.up3 = Up(4*ngf, 2*ngf, use_duse=use_duse)
        # self.up4 = Up(2*ngf, ngf, use_duse=use_duse)
        self.up1 = Up1(32 * ngf, 8 * ngf, use_duse=use_duse)
        self.up2 = Up(16 * ngf, 4 * ngf, use_duse=use_duse)
        self.up3 = Up(8 * ngf, 2 * ngf, use_duse=use_duse)
        self.up4 = Up(4 * ngf, ngf, use_duse=use_duse)
        self.outc = OutConv(ngf, self.n_classes)



    def forward(self, t1, t2, sc):
        # print('@@@@@')
        x1 = self.inc(t1)
        y1 = self.inc(t2)
        x2 = self.down1(x1)
        y2 = self.down1(y1)
        x3 = self.down2(x2)
        y3 = self.down2(y2)

        x4 = self.down3(x3)
        y4 = self.down3(y3)


        x5 = self.down4(x4)
        y5 = self.down4(y4)
        z5 = torch.cat((x5, y5), dim=1)
        z4 = torch.cat((x4, y4), dim=1)
        z3 = torch.cat((x3, y3), dim=1)
        z2 = torch.cat((x2, y2), dim=1)
        z1 = torch.cat((x1, y1), dim=1)

        x = z5
        x = self.up1(x, z4)

        x = self.up2(x, z3, sc[2])
        x = self.up3(x, z2, sc[1])
        x = self.up4(x, z1, sc[0])
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_duse=True):
        super().__init__()
        if use_duse:
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                ChannelSpatialSELayer(out_channels)
            )

        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = x.float()
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_duse=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, use_duse=use_duse)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, use_duse=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels * 2),
            nn.ReLU(inplace=True)
            )
            # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)



        # attn_scse
        # self.conv = DoubleConv(in_channels, out_channels, use_duse)
        # self.conv_scse = DoubleConv(in_channels // 2, out_channels, use_duse) if in_channels // out_channels == 4 else DoubleConv(in_channels, out_channels, use_duse)

        # att_scse2+
        self.conv = DoubleConv(in_channels, out_channels, use_duse)

        # self.conv_scse = DoubleConv(in_channels + out_channels, out_channels, use_duse)
        self.conv_scse = DoubleConv(in_channels + out_channels, out_channels, use_duse)

        # self.attn = Attention_block_3D(out_channels, out_channels, out_channels // 2)
        self.attn = Attention_block_3D(out_channels * 2, out_channels * 2, out_channels)
        self.attn_brideg = Attention_block_3D(out_channels * 2, out_channels, out_channels)

        # self.attn_axi = Axial_Attention_block_3D(in_channels // 2, 1, 1, 3)
        # self.attn_imp = ImprovedAttentionBlock(in_channels // 2, in_channels // 2, out_channels)
        # self.scse = ChannelSpatialSELayer(in_channels // 2)
    def forward(self, x1, x2, bridge=None):
        if bridge is not None:

            x = self.up(x1)


            x2 = self.attn(x, x2)
            bridge = self.attn_brideg(x, bridge)
            x = torch.cat((x2, bridge, x), dim=1)

            x = self.conv_scse(x)
            return x

        else:
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)

            return self.conv(x)

class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, use_duse=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, 2 * out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(2 * out_channels),
            nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, use_duse)

        self.conv_scse = DoubleConv(in_channels + out_channels, out_channels, use_duse)

        self.attn = Attention_block_3D(2 * out_channels, 2 * out_channels, out_channels)
        self.attn_brideg = Attention_block_3D(2 * out_channels, out_channels, out_channels)


    def forward(self, x1, x2, bridge=None):
        if bridge is not None:

            x = self.up(x1)

            x2 = self.attn(x, x2)
            bridge = self.attn_brideg(x, bridge)
            x = torch.cat((x2, bridge, x), dim=1)

            x = self.conv_scse(x)
            return x

        else:
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)

            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # return torch.sigmoid(self.conv(x))
        return torch.softmax(self.conv(x), dim=1)


class Attention_block_3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi



class Axial_Attention_block_3D(nn.Module):
    def __init__(self, dim, dim_index, heads, num_dimensions):
        super(Axial_Attention_block_3D, self).__init__()

        self.attn = AxialAttention(dim, num_dimensions, heads, dim_index=dim_index)

    def forward(self, x):
        x = self.attn(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SelfAttentionBlock, self).__init__()

        self.query_conv = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Compute queries, keys, and values
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Reshape to make the self-attention mechanism
        proj_query = proj_query.view(proj_query.size(0), -1, proj_query.size(-2) * proj_query.size(-1) * proj_query.size(-3))
        proj_key = proj_key.view(proj_key.size(0), -1, proj_key.size(-2) * proj_key.size(-1) * proj_key.size(-3))
        proj_value = proj_value.view(proj_value.size(0), -1, proj_value.size(-2) * proj_value.size(-1) * proj_value.size(-3))

        # Calculate self-attention maps
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))
        attention = self.softmax(energy)

        # Apply attention to values
        out = torch.bmm(proj_value, attention.transpose(1, 2))
        out = out.view(x.size())

        return out


class ImprovedAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ImprovedAttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.attention = SelfAttentionBlock(F_int)

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Apply the self-attention mechanism
        x1 = self.attention(x1)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """

        batch_size, num_channels, H, W, D = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()

        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1, 1))

        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b, c = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv3d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b, c)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
