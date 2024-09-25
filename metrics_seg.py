import numpy as np
import torch
from util.metrics import *
import os
import SimpleITK as sitk
import pandas as pd


def DSC_val(pred_path, label_path):
    list = os.listdir(pred_path)
    average = []
    for file in list:
        pred_file = os.path.join(pred_path, file)
        label_file = os.path.join(label_path, file)
        pred_ = sitk.ReadImage(pred_file)
        pred = sitk.GetArrayFromImage(pred_)
        pred = to_categorical(pred, 4)
        pred = np.expand_dims(pred, axis=0)
        label_ = sitk.ReadImage(label_file)
        label = sitk.GetArrayFromImage(label_)
        label = to_categorical(label, 4)
        label = np.expand_dims(label, axis=0)
        average.append(calculate_dice_average(pred, label))

    return np.mean(average)


def seg_metric_test(pred_path, label_path, csv_path):
    list = os.listdir(pred_path)
    Cerebrospinal_fluid = []
    Gray_matter = []
    White_matter = []
    Cerebrospinal_fluid_012 = []
    Gray_matter_012 = []
    White_matter_012 = []
    Cerebrospinal_fluid_1236 = []
    Gray_matter_1236 = []
    White_matter_1236 = []


    csf_asd = []
    gm_asd = []
    wm_asd = []
    csf_asd_012 = []
    gm_asd_012 = []
    wm_asd_012 = []
    csf_asd_1236 = []
    gm_asd_1236 = []
    wm_asd_1236 = []

    name_012 = []
    name_1236 = []
    dice_AVE = []
    asd_AVE = []
    for i, file in enumerate(list):
        pred_file = os.path.join(pred_path, file)
        label_file = os.path.join(label_path, file)
        pred_ = sitk.ReadImage(pred_file)
        pred = sitk.GetArrayFromImage(pred_)
        print(pred.shape)
        pred = to_categorical(pred, 4)
        pred = np.expand_dims(pred, axis=0)
        label_ = sitk.ReadImage(label_file)
        label = sitk.GetArrayFromImage(label_)
        print(label.shape)
        label = to_categorical(label, 4)
        label = np.expand_dims(label, axis=0)
        ave_dice = calculate_dice_average(pred, label)
        dice_AVE.append(ave_dice)
        ave_asd = asd_ave(pred, label)
        asd_AVE.append(ave_asd)
        if int(file[4:6]) < 13:
            name_012.append(file)
            a, b, c = calculate_dice(pred, label)
            x, y, z = asd(pred, label)
            Cerebrospinal_fluid_012.append(a)
            Gray_matter_012.append(b)
            White_matter_012.append(c)
            csf_asd_012.append(x)
            gm_asd_012.append(y)
            wm_asd_012.append(z)
        elif int(file[4:6]) > 12:
            name_1236.append(file)
            a, b, c = calculate_dice(pred, label)
            x, y, z = asd(pred, label)
            Cerebrospinal_fluid_1236.append(a)
            Gray_matter_1236.append(b)
            White_matter_1236.append(c)
            csf_asd_1236.append(x)
            gm_asd_1236.append(y)
            wm_asd_1236.append(z)


        X, Y, Z = asd(pred, label)

        csf_asd.append(X)
        gm_asd.append(Y)
        wm_asd.append(Z)
        A, B, C = calculate_dice(pred, label)
        Cerebrospinal_fluid.append(A)
        Gray_matter.append(B)
        White_matter.append(C)
        # q, w, e = HD_95(pred, label)
        # csf_hd.append(q)
        # gm_hd.append(w)
        # wm_hd.append(e)
        # csf_asd[i], gm_asd[i], wm_asd[i] = asd(pred, label)
        # csf_hd[i], gm_hd[i], wm_hd[i] = HD_95(pred, label)

    Cerebrospinal_fluid_ave = np.mean(Cerebrospinal_fluid)
    Gray_matter_ave = np.mean(Gray_matter)
    White_matter_ave = np.mean(White_matter)
    dice_average = np.mean(dice_AVE)

    asd_average = np.mean(asd_AVE)
    csf_ave = np.mean(csf_asd)
    gm_ave = np.mean(gm_asd)
    wm_ave = np.mean(wm_asd)

    print('Dice: Cerebrospinal_fluid:%s  Gray_matter:%s  White_matter:%s  Overview:%s' % (Cerebrospinal_fluid_ave, Gray_matter_ave, White_matter_ave, dice_average))
    print('ASD: Cerebrospinal_fluid:%s  Gray_matter:%s  White_matter:%s  Overview:%s' % (csf_ave, gm_ave, wm_ave, asd_average))


    df_dice_012 = pd.DataFrame({'name': name_012, 'Cerebrospinal_fluid': Cerebrospinal_fluid_012, 'Gray_matter': Gray_matter_012,
                            'White_matter': White_matter_012})
    df_dice_012.to_csv(os.path.join(csv_path, 'ABCL-Net_dice_0-12.csv'), index=False)
    df_dice_1236 = pd.DataFrame({'name': name_1236, 'Cerebrospinal_fluid': Cerebrospinal_fluid_1236, 'Gray_matter': Gray_matter_1236,
         'White_matter': White_matter_1236})
    df_dice_1236.to_csv(os.path.join(csv_path, 'ABCL-Net_dice_12-36.csv'), index=False)
    df_ASD_012 = pd.DataFrame({'name': name_012, 'Cerebrospinal_fluid': csf_asd_012, 'Gray_matter': gm_asd_012, 'White_matter': wm_asd_012})
    df_ASD_012.to_csv(os.path.join(csv_path, 'ABCL-Net_ASD_0-12.csv'), index=False)
    df_ASD_1236 = pd.DataFrame(
        {'name': name_1236, 'Cerebrospinal_fluid': csf_asd_1236, 'Gray_matter': gm_asd_1236, 'White_matter': wm_asd_1236})
    df_ASD_1236.to_csv(os.path.join(csv_path, 'ABCL-Net_ASD_12-36.csv'), index=False)


def gen_metric_test(gen_path, ori_path, csv_path):
    list_gen = os.listdir(gen_path)
    psnr = []
    ssim = []
    psnr_012 = []
    psnr_1236 = []
    ssim_012 = []
    ssim_1236 = []

    name_012 = []
    name_1236 = []
    for j, file in enumerate(list_gen):

        gen_T2 = sitk.ReadImage(os.path.join(gen_path, file))
        gen_T2 = sitk.GetArrayFromImage(gen_T2)
        ori_T2 = sitk.ReadImage(os.path.join(ori_path, file))
        ori_T2 = sitk.GetArrayFromImage(ori_T2)
        gen_T2 = (gen_T2 - np.min(gen_T2)) / (np.max(gen_T2) - np.min(gen_T2)) * 255
        ori_T2 = (ori_T2 - np.min(ori_T2)) / (np.max(ori_T2) - np.min(ori_T2)) * 255
        psnr.append(calculate_psnr_test(ori_T2, gen_T2))
        ssim.append(calculate_ssim(ori_T2, gen_T2))
        if int(file[4:6]) < 13:
            name_012.append(file)
            psnr_012.append(calculate_psnr_test(ori_T2, gen_T2))
            ssim_012.append(calculate_ssim(ori_T2, gen_T2))
        elif int(file[4:6]) > 12:
            name_1236.append(file)
            psnr_1236.append(calculate_psnr_test(ori_T2, gen_T2))
            ssim_1236.append(calculate_ssim(ori_T2, gen_T2))


    psnr_ave = np.mean(psnr)
    ssim_ave = np.mean(ssim)
    print('psnr:', psnr_ave)
    print('ssim', ssim_ave)

    df_gen_012 = pd.DataFrame({'name': name_012, 'psnr': psnr_012, 'ssim': ssim_012})
    df_gen_012.to_csv(os.path.join(csv_path, 'ABCL-Net_gen_0-12.csv'), index=False)
    df_gen_1236 = pd.DataFrame({'name': name_1236, 'psnr': psnr_1236, 'ssim': ssim_1236})
    df_gen_1236.to_csv(os.path.join(csv_path, 'ABCL-Net_gen_12-36.csv'), index=False)

if __name__ == "__main__":
    pred_path = ''
    label_path = ''
    out_path = ''
    seg_metric_test(pred_path, label_path, out_path)
