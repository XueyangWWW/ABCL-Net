import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import SimpleITK as sitk
import os
import pandas as pd
import surface_distance as surfdist


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()

    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]

    categorical = np.zeros((num_classes, n), dtype='float32')

    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)

    return categorical

def calculate_psnr_train(image1, image2):
    difference = image1 - image2
    mse = np.mean(np.square(difference))
    PSNR = 10 * np.log10(1 / mse)
    return PSNR

def calculate_psnr_test(target, pred):
    score = psnr(target, pred, data_range=255)
    return score

def calculate_ssim(pred, target):

    pred = pred.squeeze()
    target = target.squeeze()
    score = ssim(pred, target, data_range=255, channel_axis=True)
    return score

def dice_ratio(pred, label):

    return np.sum(pred[label == 1]) * 2.0 / (np.sum(pred) + np.sum(label))


def calculate_dice(pred, label):

    A_pred = pred[:, 0, :, :, :]            # background
    B_pred = pred[:, 1, :, :, :]            # Cerebrospinal fluid
    C_pred = pred[:, 2, :, :, :]            # Gray matter
    D_pred = pred[:, 3, :, :, :]            # White matter

    A_label = label[:, 0, :, :, :]
    B_label = label[:, 1, :, :, :]
    C_label = label[:, 2, :, :, :]
    D_label = label[:, 3, :, :, :]

    A_dr = dice_ratio(A_pred, A_label)
    B_dr = dice_ratio(B_pred, B_label)
    C_dr = dice_ratio(C_pred, C_label)
    D_dr = dice_ratio(D_pred, D_label)

    return B_dr, C_dr, D_dr

def calculate_dice_average(pred, label):

    A_pred = pred[:, 0, :, :, :]            # background
    B_pred = pred[:, 1, :, :, :]            # Cerebrospinal fluid
    C_pred = pred[:, 2, :, :, :]            # Gray matter
    D_pred = pred[:, 3, :, :, :]            # White matter

    A_label = label[:, 0, :, :, :]
    B_label = label[:, 1, :, :, :]
    C_label = label[:, 2, :, :, :]
    D_label = label[:, 3, :, :, :]

    A_dr = dice_ratio(A_pred, A_label)
    B_dr = dice_ratio(B_pred, B_label)
    C_dr = dice_ratio(C_pred, C_label)
    D_dr = dice_ratio(D_pred, D_label)

    # avg = np.mean([A_dr, B_dr, C_dr, D_dr])
    avg_ = np.mean([B_dr, C_dr, D_dr])
    return avg_



def HausdorffDistance(predict, label, index=1):
    predict = (predict == index).astype(np.uint8)
    label = (label == index).astype(np.uint8)
    predict_sum = predict.sum()
    label_sum = label.sum()
    if predict_sum != 0 and label_sum != 0:
        mask1 = sitk.GetImageFromArray(predict, isVector=False)
        mask2 = sitk.GetImageFromArray(label, isVector=False)
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(mask1, mask2)
        result1 = hausdorff_distance_filter.GetHausdorffDistance()
        result2 = hausdorff_distance_filter.GetAverageHausdorffDistance()
        result = result1, result2
    elif predict_sum != 0 and label_sum == 0:
        result = 'FP', 'FP'
    elif predict_sum == 0 and label_sum != 0:
        result = 'FN', 'FN'
    else:
        result = 'TN', 'TN'
    return result


def Getcontour(img):

    image = sitk.GetImageFromArray(img.astype(np.uint8), isVector=False)
    filter = sitk.SimpleContourExtractorImageFilter()
    image = filter.Execute(image)
    image = sitk.GetArrayFromImage(image)
    return image.astype(np.uint8)


def HDAVD(n_classes, pred_dir, gt_dir, csv_path):
    pred_filenames = os.listdir(pred_dir)
    hauAve = np.zeros(shape=(n_classes, len(pred_filenames)), dtype=np.float32)
    hau = np.zeros(shape=(n_classes, len(pred_filenames)), dtype=np.float32)

    save_name_HD = 'Acc_HD.csv'
    save_name_AVD = 'Acc_AVD.csv'

    csf_HD = []
    gm_HD = []
    wm_HD = []
    csf_AVD = []
    gm_AVD = []
    wm_AVD = []
    for i in range(len(pred_filenames)):
        name = pred_filenames[i]

        groundtruth = sitk.ReadImage(os.path.join(gt_dir, name))
        originSpacing = groundtruth.GetSpacing()
        originSize = groundtruth.GetSize()
        newSize = [int(round(originSize[0] * originSpacing[0])), int(round(originSize[1] * originSpacing[1])),
                   int(round(originSize[2] * originSpacing[2]))]
        newSpacing = [1, 1, 1]

        predict = sitk.ReadImage(os.path.join(pred_dir, name))
        predict.SetSpacing([originSpacing[0], originSpacing[0], originSpacing[0]])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(groundtruth)
        resampler.SetSize(newSize)
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        groundtruth = resampler.Execute(groundtruth)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(predict)
        resampler.SetSize(newSize)
        resampler.SetOutputSpacing(newSpacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        predict = resampler.Execute(predict)

        predict = sitk.GetArrayFromImage(predict)
        predict = to_categorical(predict, num_classes=n_classes)

        groundtruth = sitk.GetArrayFromImage(groundtruth)
        groundtruth = to_categorical(groundtruth, num_classes=n_classes)

        for c in range(n_classes):
            predict_suf = Getcontour(predict[c])
            label_suf = Getcontour(groundtruth[c])
            HD_AVD = HausdorffDistance(predict_suf, label_suf)
            if HD_AVD[0] == 'FN' or HD_AVD[0] == 'FP' or HD_AVD[0] == 'TN':
                predict_suf = Getcontour(np.ones_like(predict[c]))
                hau[c, i], hauAve[c, i] = HausdorffDistance(predict_suf, label_suf)
            else:
                hau[c, i], hauAve[c, i] = HD_AVD[0], HD_AVD[1]
        csf_HD.append(hau[1, i])
        gm_HD.append(hau[2, i])
        wm_HD.append(hau[3, i])
        csf_AVD.append(hauAve[1, i])
        gm_AVD.append(hauAve[2, i])
        wm_AVD.append(hauAve[3, i])
        # print(name, hau[1:, i], hauAve[1:, i])
    print('HD: Cerebrospinal_fluid:%s  Gray_matter:%s  White_matter:%s' % (np.mean(csf_HD), np.mean(gm_HD), np.mean(wm_HD)))
    print('AVD: Cerebrospinal_fluid:%s  Gray_matter:%s  White_matter:%s' % (np.mean(csf_AVD), np.mean(gm_AVD), np.mean(wm_AVD)))
    save_df = csv_path
    df_AVD = pd.DataFrame({'Name': pred_filenames, 'csf': hauAve[1], 'gm': hauAve[2], 'wm': hauAve[3]})
    df_AVD.to_csv(save_name_AVD, index=False)
    # df_HD = pd.DataFrame({'Cerebrospinal_fluid': [np.mean(csf_HD)], 'Gray_matter': [np.mean(gm_HD)], 'White_matter': [np.mean(wm_HD)]})
    df_HD = pd.DataFrame({'Name': pred_filenames, 'csf': hau[1], 'gm': hau[2], 'wm': hau[3]})
    df_HD.to_csv(os.path.join(save_df, save_name_HD), index=False)


# def HD_95(pred, label):
#     A_pred = pred[:, 0, :, :, :].squeeze().astype(np.bool)  # background
#     B_pred = pred[:, 1, :, :, :].squeeze().astype(np.bool)  # Cerebrospinal fluid
#     C_pred = pred[:, 2, :, :, :].squeeze().astype(np.bool)  # Gray matter
#     D_pred = pred[:, 3, :, :, :].squeeze().astype(np.bool)  # White matter
#
#     A_label = label[:, 0, :, :, :].squeeze().astype(np.bool)
#     B_label = label[:, 1, :, :, :].squeeze().astype(np.bool)
#     C_label = label[:, 2, :, :, :].squeeze().astype(np.bool)
#     D_label = label[:, 3, :, :, :].squeeze().astype(np.bool)
#
#     A_surface_distances = surfdist.compute_surface_distances(A_label, A_pred, spacing_mm=(1.0, 1.0, 1.0))
#     A_hd_dist_95 = surfdist.compute_robust_hausdorff(A_surface_distances, 95)
#     B_surface_distances = surfdist.compute_surface_distances(B_label, B_pred, spacing_mm=(1.0, 1.0, 1.0))
#     B_hd_dist_95 = surfdist.compute_robust_hausdorff(B_surface_distances, 95)
#     C_surface_distances = surfdist.compute_surface_distances(C_label, C_pred, spacing_mm=(1.0, 1.0, 1.0))
#     C_hd_dist_95 = surfdist.compute_robust_hausdorff(C_surface_distances, 95)
#     D_surface_distances = surfdist.compute_surface_distances(D_label, D_pred, spacing_mm=(1.0, 1.0, 1.0))
#     D_hd_dist_95 = surfdist.compute_robust_hausdorff(D_surface_distances, 95)
#
#     return B_hd_dist_95, C_hd_dist_95, D_hd_dist_95


def asd(pred, label):
    A_pred = pred[:, 0, :, :, :].squeeze().astype(bool)  # background
    B_pred = pred[:, 1, :, :, :].squeeze().astype(bool)  # Cerebrospinal fluid
    C_pred = pred[:, 2, :, :, :].squeeze().astype(bool)  # Gray matter
    D_pred = pred[:, 3, :, :, :].squeeze().astype(bool)  # White matter

    A_label = label[:, 0, :, :, :].squeeze().astype(bool)
    B_label = label[:, 1, :, :, :].squeeze().astype(bool)
    C_label = label[:, 2, :, :, :].squeeze().astype(bool)
    D_label = label[:, 3, :, :, :].squeeze().astype(bool)

    A_surface_distances = surfdist.compute_surface_distances(A_label, A_pred, spacing_mm=(1.0, 1.0, 1.0))
    A_surf_dist = surfdist.compute_average_surface_distance(A_surface_distances)
    B_surface_distances = surfdist.compute_surface_distances(B_label, B_pred, spacing_mm=(1.0, 1.0, 1.0))
    B_surf_dist = surfdist.compute_average_surface_distance(B_surface_distances)
    C_surface_distances = surfdist.compute_surface_distances(C_label, C_pred, spacing_mm=(1.0, 1.0, 1.0))
    C_surf_dist = surfdist.compute_average_surface_distance(C_surface_distances)
    D_surface_distances = surfdist.compute_surface_distances(D_label, D_pred, spacing_mm=(1.0, 1.0, 1.0))
    D_surf_dist = surfdist.compute_average_surface_distance(D_surface_distances)

    A_avg_surf_dist = (A_surf_dist[0] + A_surf_dist[1]) / 2
    B_avg_surf_dist = (B_surf_dist[0] + B_surf_dist[1]) / 2
    C_avg_surf_dist = (C_surf_dist[0] + C_surf_dist[1]) / 2
    D_avg_surf_dist = (D_surf_dist[0] + D_surf_dist[1]) / 2
    return B_avg_surf_dist, C_avg_surf_dist, D_avg_surf_dist

def asd_ave(pred, label):
    A_pred = pred[:, 0, :, :, :].squeeze().astype(bool)  # background
    B_pred = pred[:, 1, :, :, :].squeeze().astype(bool)  # Cerebrospinal fluid
    C_pred = pred[:, 2, :, :, :].squeeze().astype(bool)  # Gray matter
    D_pred = pred[:, 3, :, :, :].squeeze().astype(bool)  # White matter

    A_label = label[:, 0, :, :, :].squeeze().astype(bool)
    B_label = label[:, 1, :, :, :].squeeze().astype(bool)
    C_label = label[:, 2, :, :, :].squeeze().astype(bool)
    D_label = label[:, 3, :, :, :].squeeze().astype(bool)

    A_surface_distances = surfdist.compute_surface_distances(A_label, A_pred, spacing_mm=(1.0, 1.0, 1.0))
    A_surf_dist = surfdist.compute_average_surface_distance(A_surface_distances)
    B_surface_distances = surfdist.compute_surface_distances(B_label, B_pred, spacing_mm=(1.0, 1.0, 1.0))
    B_surf_dist = surfdist.compute_average_surface_distance(B_surface_distances)
    C_surface_distances = surfdist.compute_surface_distances(C_label, C_pred, spacing_mm=(1.0, 1.0, 1.0))
    C_surf_dist = surfdist.compute_average_surface_distance(C_surface_distances)
    D_surface_distances = surfdist.compute_surface_distances(D_label, D_pred, spacing_mm=(1.0, 1.0, 1.0))
    D_surf_dist = surfdist.compute_average_surface_distance(D_surface_distances)

    A_avg_surf_dist = (A_surf_dist[0] + A_surf_dist[1]) / 2
    B_avg_surf_dist = (B_surf_dist[0] + B_surf_dist[1]) / 2
    C_avg_surf_dist = (C_surf_dist[0] + C_surf_dist[1]) / 2
    D_avg_surf_dist = (D_surf_dist[0] + D_surf_dist[1]) / 2
    ave = np.mean([B_avg_surf_dist, C_avg_surf_dist, D_avg_surf_dist])
    return ave
