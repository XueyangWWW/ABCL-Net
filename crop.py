import os

import SimpleITK as sitk
import numpy as np


# '''
#       Crop images into 160x160x160
# '''

def cut_edge(data):
    '''Cuts zero edge for a 3D image.
    Args:
        data: A 3D image, [Depth, Height, Width, 1].
    Returns:
        original_shape: [Depth, Height, Width]
        cut_size: A list of six integers [Depth_s, Depth_e, Height_s, Height_e, Width_s, Width_e]
    '''

    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    original_shape = [D, H, W]
    cut_size = [int(D_s), int(D_e + 1), int(H_s), int(H_e + 1), int(W_s), int(W_e + 1)]

    return (original_shape, cut_size)


def fixed_crop(data, img_size):
    D, H, W = data.shape

    a = (D - img_size) / 2
    b = (H - img_size) / 2
    c = (W - img_size) / 2

    cut_size = [int(a), int(D - a), int(b), int(H - b), int(c), int(W - c)]
    return cut_size

def crop_syn_t2(syn_t2, ori_t2, dir_syn_t2, dir_ori_t2):
    if not os.path.exists(dir_syn_t2):
        os.makedirs(dir_syn_t2)
    if not os.path.exists(dir_ori_t2):
        os.makedirs(dir_ori_t2)
    file_t2 = os.listdir(ori_t2)
    for file in file_t2:
        frame = os.path.join(ori_t2, file)
        frame1 = os.path.join(syn_t2, file)
        img_t2 = sitk.ReadImage(frame)
        img_t2_syn = sitk.ReadImage(frame1)
        t2 = sitk.GetArrayFromImage(img_t2)
        t2_syn = sitk.GetArrayFromImage(img_t2_syn)
        (oS, cS) = cut_edge(t2)
        T2 = t2[cS[0]:cS[1], cS[2]:cS[3], cS[4]:cS[5]]
        T2_syn = t2_syn[cS[0]:cS[1], cS[2]:cS[3], cS[4]:cS[5]]
        T2 = sitk.GetImageFromArray(T2)
        sitk.WriteImage(T2, os.path.join(dir_ori_t2, file))
        T2_syn = sitk.GetImageFromArray(T2_syn)
        sitk.WriteImage(T2_syn, os.path.join(dir_syn_t2, file))

if __name__ == '__main__':
    file_path_T1 = None    # origin T1 path
    file_path_T2 = None    # origin T2 path
    file_path_gt = None    # origin label path
    file_dir_T1 = None    # cut T1 path
    file_dir_T2 = None    # cut T2 path
    file_dir_gt = None    # cut label path

    assert file_path_T1 is not None, "file_path_T1 CANNOT BE NONE"
    assert file_path_T2 is not None, "file_path_T2 CANNOT BE NONE"
    assert file_dir_gt is not None, "file_dir_gt CANNOT BE NONE"
    assert file_dir_T1 is not None, "file_dir_T1 CANNOT BE NONE"
    assert file_dir_T2 is not None, "file_dir_T2 CANNOT BE NONE"
    assert file_dir_gt is not None, "file_dir_gt CANNOT BE NONE"


    file_T1 = sorted(os.listdir(file_path_T1))
    for file in file_T1:

        fname_T1 = file_path_T1+file
        fname_T2 = file_path_T2+file.replace('T1', 'T2')
        fname_gt = file_path_gt+file.replace('T1', 'Seg')
        img_1 = sitk.ReadImage(fname_T1)
        spacing = img_1.GetSpacing()
        origin = img_1.GetOrigin()
        direction = img_1.GetDirection()
        img_T1 = sitk.GetArrayFromImage(img_1)
        img_2 = sitk.ReadImage(fname_T2)
        img_T2 = sitk.GetArrayFromImage(img_2)
        img_3 = sitk.ReadImage(fname_gt)
        img_gt = sitk.GetArrayFromImage(img_3)
        print(img_T1.shape)


        cs = fixed_crop(img_T1, 160)     #固定距离裁剪

        T1 = img_T1[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]
        T2 = img_T2[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]
        gt = img_gt[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]

        img_t1 = sitk.GetImageFromArray(T1)
        img_t2 = sitk.GetImageFromArray(T2)
        img_GT = sitk.GetImageFromArray(gt)
        img_t1.SetSpacing(spacing)
        img_t1.SetOrigin(origin)
        img_t1.SetDirection(direction)
        img_t2.SetSpacing(spacing)
        img_t2.SetOrigin(origin)
        img_t2.SetDirection(direction)
        img_GT.SetSpacing(spacing)
        img_GT.SetOrigin(origin)
        img_GT.SetDirection(direction)
        sitk.WriteImage(img_t1, file_dir_T1+file)
        sitk.WriteImage(img_t2, file_dir_T2+file.replace('T1', 'T2'))
        sitk.WriteImage(img_GT, file_dir_gt+file.replace('T1', 'Seg'))

