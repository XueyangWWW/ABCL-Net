import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
from torch.autograd import Variable
from util.metrics import calculate_psnr_train, calculate_ssim, calculate_dice_average
from .base_model import BaseModel
from . import networks_DE4

import itertools
import util.util as util
from models.pre_r3d_18 import Res3D
from util.image_pool import ImagePool
from models.networks_DE4 import GANLoss, feature_loss, discriminate
from util.loss_functions import mix_loss

class GEN2SEGModel_TRAIN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'MSE', 'Seg']
        self.metric_names = ['PSNR', 'DICE']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'seg_real_A', 'seg_fake_B']
        if self.isTrain:
            self.model_names = ['G', 'D', 'S']
        else:
            self.model_names = ['S']

        # define networks (both generator and discriminator)
        self.netG = networks_DE4.define_G(opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        self.ext_discriminator = Res3D().cuda().to(self.device)

        self.netD = networks_DE4.define_D(2, 64, 2, 'instance3D', False, 3, True, [0]).to(self.device)
        self.netS = networks_DE4.define_S(opt.input_nc, opt.output_nc_seg, opt.ngf, opt.netS, opt.normS,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                      opt.no_antialias_up, self.gpu_ids, opt).to(self.device)


        # define loss functions
        self.criterionGAN = GANLoss(use_lsgan=not False, tensor=torch.cuda.FloatTensor)
        self.mix_loss = mix_loss()
        self.mse = torch.nn.MSELoss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netS.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.fake_pool = ImagePool(0)
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)



    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()

        self.loss_G.backward(retain_graph=False)
        self.optimizer_G.step()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()



    def set_input(self, input):

        self.real_A = torch.unsqueeze(input['img_A'], dim=1).to(self.device)
        self.real_B = torch.unsqueeze(input['img_B'], dim=1).to(self.device)

        self.seg_real_A = input['Seg'].to(self.device)
        # self.seg_real_A_ori = torch.unsqueeze(input['Seg_ori'], 1).to(self.device)


        self.image_paths = input['A_paths']

    def start_validating(self, input):

        self.real_A = torch.unsqueeze(input['img_A'].to(self.device), dim=1)
        self.real_B = torch.unsqueeze(input['img_B'].to(self.device), dim=1)

        self.seg_real_A = input['Seg'].to(self.device)
        # self.seg_real_A_ori = torch.unsqueeze(input['Seg_ori'], 1).to(self.device)


        self.image_paths = input['A_paths']

        self.forward()
        return self.metric_DICE, self.metric_PSNR

    def forward(self):
        # self.real = self.real_A

        self.fake = self.netG(self.real_A)[0]
        sc = self.netG(self.real_A)[1]

        t1 = self.real_A
        t2 = self.fake
        # t2 = self.real_B
        self.fake_B = self.fake
        self.seg_fake_B = self.netS(t1, t2, sc)


        # Calculate GAN loss for the discriminator
        self.pred_fake_pool = discriminate(self.netD, self.fake_pool, self.real_A, self.fake, use_pool=True)
        self.loss_D_fake = self.criterionGAN(self.pred_fake_pool, False).mean()

        # Real Detection and Loss
        self.pred_real = discriminate(self.netD, self.fake_pool, self.real_A, self.real_B)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()
        self.pred_fake = self.netD.forward(torch.cat((self.real_A, self.fake), dim=1))
        self.loss_G_GAN = self.criterionGAN(self.pred_fake, True).mean()


        self.loss_G_GAN_Feat, self.loss_G_GAN_Feat_ext = feature_loss(self.real_B, self.fake, self.pred_real, self.pred_fake,
                                                                      self.ext_discriminator)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_MSE = self.mse(self.fake.float(), self.real_B.float())

        self.seg_output = self.seg_fake_B

        self.loss_Seg = self.mix_loss(self.seg_real_A, self.seg_output)
        self.loss_G = self.loss_G_GAN + 2 * self.loss_Seg + self.loss_MSE * 100.0 + \
                      self.loss_G_GAN_Feat_ext * 10.0 + self.loss_G_GAN_Feat * 10.0

        self.metric_PSNR = calculate_psnr_train(self.real_B.cpu().detach().numpy(), self.fake.cpu().detach().numpy())
        # self.metric_SSIM = calculate_ssim(self.real_B.cpu().detach().numpy(), self.fake_B.cpu().detach().numpy(), 1)
        self.metric_DICE = calculate_dice_average(self.seg_fake_B.cpu().detach().numpy(), self.seg_real_A.cpu().detach().numpy())


class GEN2SEGModel_TEST(BaseModel):
    def __init__(self, opt):
        assert (not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.netS = networks_DE4.define_S(opt.input_nc_seg, opt.output_nc_seg, opt.ngf, opt.netS, opt.normS,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                      opt.no_antialias_up, self.gpu_ids, opt).to(self.device)
        self.netG = networks_DE4.define_G(opt.init_type, opt.init_gain, self.gpu_ids).to(self.device)
        self.load_network_G(self.netG, 'G', opt.which_epoch_G)
        self.load_network_S(self.netS, 'S', opt.which_epoch_S)

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def set_input(self, input):

        # self.real_A = torch.unsqueeze(input['A_img'].to(self.device), dim=1)
        # self.real_B = torch.unsqueeze(input['B_img'].to(self.device), dim=1)
        self.real_A = torch.unsqueeze(input['A_img'], dim=1).to(self.device)
        # self.real_B = torch.unsqueeze(input['B_img'], dim=1).to(self.device)
        # self.seg_real_B_gt = torch.unsqueeze(input['Seg_img'], dim=1).to(self.device)
        # self.Seg_real_B_gt = self.seg_real_B_gt.unsqueeze(1)
        self.image_paths = input['A_paths']

    def test(self):
    #   input patches
        self.real_A = Variable(self.real_A)

        # self.real_B = Variable(self.real_B)

        b = self.netG(self.real_A)
        self.fake_B = b[0]
        sc = b[1]

        self.fake_B = Variable(self.fake_B)

        t1 = self.real_A
        t2 = self.fake_B
        # self.input_seg = torch.cat((self.fake_B, self.real_A), dim=1)
        # self.seg_fake_B = self.netS(self.input_seg, sc)


        # self.seg_fake_B = self.netS(self.real_A, self.real_B, sc)  # T1+T2

        self.seg_fake_B = self.netS(t1, t2, sc)  # T1+T2*

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        # real_A = util.tensor2im(self.real_A.data)
        # real_B = util.tensor2im(self.real_B.data)
        fake_B = util.tensor2im(self.fake_B.data)

        # seg_fake_B = self.seg_fake_B.data[:, -1:, :, :]
        # seg_fake_B[seg_fake_B >= 0.5] = 1
        # seg_fake_B[seg_fake_B < 0.5] = 0
        seg_fake_B = util.tensor2seg_output_test(self.seg_fake_B.data)

        # seg_fake_B = util.tensor2seg_test(self.seg_fake_B.data)
        # seg_real_B_gt = self.seg_real_B_gt.data[:, -1:, :, :]
        # seg_real_B_gt[seg_real_B_gt >= 0.5] = 1
        # seg_real_B_gt[seg_real_B_gt < 0.5] = 0
        # seg_real_B_gt = util.tensor2seg_target_test(self.seg_real_B_gt.data)
        # seg_real_B_gt = util.tensor2seg_test(self.seg_real_B_gt.data)
        # seg_real_B_gt = util.tensor2seg_test(torch.max(self.seg_real_B_gt.data, dim=1, keepdim=True)[1])

        # return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('seg_fake_B', seg_fake_B),
        #                     ('gt_real_B_seg', seg_real_B_gt)])
        return OrderedDict([('fake_B', fake_B), ('seg_fake_B', seg_fake_B)])
