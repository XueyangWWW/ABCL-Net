import os
import SimpleITK as sitk
import numpy as np
import argparse

def cut_edge(data):
    """
    Automatically trim zero-valued edges from a 3D image.
    Returns the bounding box that contains all non-zero voxels.
    """
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

    if D_s > D_e or H_s > H_e or W_s > W_e:
        # No non-zero voxels found, use the full image
        D_s, D_e, H_s, H_e, W_s, W_e = 0, D - 1, 0, H - 1, 0, W - 1

    cut_size = [int(D_s), int(D_e + 1), int(H_s), int(H_e + 1), int(W_s), int(W_e + 1)]
    return cut_size

def pad_to_multiple(data, multiple=64):
    """
    Pad the data so that each dimension is a multiple of 'multiple'.
    """
    D, H, W = data.shape
    D_pad = (multiple - (D % multiple)) if D % multiple != 0 else 0
    H_pad = (multiple - (H % multiple)) if H % multiple != 0 else 0
    W_pad = (multiple - (W % multiple)) if W % multiple != 0 else 0

    pad_before_D = D_pad // 2
    pad_after_D = D_pad - pad_before_D

    pad_before_H = H_pad // 2
    pad_after_H = H_pad - pad_before_H

    pad_before_W = W_pad // 2
    pad_after_W = W_pad - pad_before_W

    data_padded = np.pad(
        data, ((pad_before_D, pad_after_D),
               (pad_before_H, pad_after_H),
               (pad_before_W, pad_after_W)),
        mode='constant', constant_values=0)
    return data_padded, (pad_before_D, pad_before_H, pad_before_W)

def extract_patches_64(data, data_t2, data_seg, patch_size=64, stride_d=64, stride_h=64, stride_w=64,
                       base_name='Case_001_T1w', des='patch_T1w', des1='patch_T2w', des2='patch_Seg', des_zero='patch_Zero'):
    """
    Extract patches of size patch_size^3 from the given 3D arrays (T1w, T2w, Seg).
    Stride in each dimension can be specified.
    Non-zero patches are saved to des/des1/des2, zero patches to des_zero.
    """
    os.makedirs(des, exist_ok=True)
    os.makedirs(des1, exist_ok=True)
    os.makedirs(des2, exist_ok=True)
    os.makedirs(des_zero, exist_ok=True)

    D, H, W = data.shape
    patch_idx = 1
    count_zero = 0

    # The order of loops determines the scanning order.
    # For each (H, W) position, we cover all patches along D with specified strides.
    for h_start in range(0, H - patch_size + 1, stride_h):
        for w_start in range(0, W - patch_size + 1, stride_w):
            for d_start in range(0, D - patch_size + 1, stride_d):
                d_end = d_start + patch_size
                h_end = h_start + patch_size
                w_end = w_start + patch_size

                patch_arr = data[d_start:d_end, h_start:h_end, w_start:w_end]
                patch_arr1 = data_t2[d_start:d_end, h_start:h_end, w_start:w_end]
                patch_arr2 = data_seg[d_start:d_end, h_start:h_end, w_start:w_end]

                if patch_idx < 10:
                    patch_name = base_name.split('.nii.gz')[0] + '_00' + str(patch_idx) + '.nii.gz'
                elif patch_idx < 100:
                    patch_name = base_name.split('.nii.gz')[0] + '_0' + str(patch_idx) + '.nii.gz'
                else:
                    patch_name = base_name.split('.nii.gz')[0] + '_' + str(patch_idx) + '.nii.gz'

                patch = sitk.GetImageFromArray(patch_arr)
                patch1 = sitk.GetImageFromArray(patch_arr1)
                patch2 = sitk.GetImageFromArray(patch_arr2)

                patch_name_t2 = patch_name.replace('T1w', 'T2w')
                patch_name_seg = patch_name.replace('T1w', 'Seg')

                if np.max(patch_arr) != 0 and np.max(patch_arr1) != 0 and np.max(patch_arr2) != 0:
                    sitk.WriteImage(patch, os.path.join(des, patch_name))
                    sitk.WriteImage(patch1, os.path.join(des1, patch_name_t2))
                    sitk.WriteImage(patch2, os.path.join(des2, patch_name_seg))
                else:
                    sitk.WriteImage(patch, os.path.join(des_zero, patch_name))
                    sitk.WriteImage(patch1, os.path.join(des_zero, patch_name_t2))
                    sitk.WriteImage(patch2, os.path.join(des_zero, patch_name_seg))
                    count_zero += 1

                patch_idx += 1

    print(str(patch_idx - 1) + " patches extracted!")
    print(count_zero, 'patches are all zero')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 3D patches from MRI data.")
    parser.add_argument('--base_dir', type=str, default='MRI_data', help='Base directory containing MRI data')
    parser.add_argument('--patch_out_base', type=str, default='patch_data_cutedge', help='Output directory for patches')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size (cube)')
    parser.add_argument('--stride_d', type=int, default=64, help='Stride along D dimension')
    parser.add_argument('--stride_h', type=int, default=64, help='Stride along H dimension')
    parser.add_argument('--stride_w', type=int, default=64, help='Stride along W dimension')

    args = parser.parse_args()

    base_dir = args.base_dir
    file_path_T1 = os.path.join(base_dir, 'T1w')
    file_path_T2 = os.path.join(base_dir, 'T2w')
    file_path_gt = os.path.join(base_dir, 'Tissue')

    patch_out_base = args.patch_out_base
    dir_T1_patch = os.path.join(patch_out_base, 'T1w')
    dir_T2_patch = os.path.join(patch_out_base, 'T2w')
    dir_Seg_patch = os.path.join(patch_out_base, 'Seg')
    dir_0_patch = os.path.join(patch_out_base, 'Zero')

    os.makedirs(dir_T1_patch, exist_ok=True)
    os.makedirs(dir_T2_patch, exist_ok=True)
    os.makedirs(dir_Seg_patch, exist_ok=True)
    os.makedirs(dir_0_patch, exist_ok=True)

    list_T1 = sorted([f for f in os.listdir(file_path_T1) if f.endswith('.nii.gz')])

    for file in list_T1:
        print("Processing:", file)
        fname_T1 = os.path.join(file_path_T1, file)
        fname_T2 = os.path.join(file_path_T2, file.replace('T1w', 'T2w'))
        fname_gt = os.path.join(file_path_gt, file.replace('T1w', 'Seg'))

        img_1 = sitk.ReadImage(fname_T1)
        img_T1 = sitk.GetArrayFromImage(img_1)

        img_2 = sitk.ReadImage(fname_T2)
        img_T2 = sitk.GetArrayFromImage(img_2)

        img_3 = sitk.ReadImage(fname_gt)
        img_gt = sitk.GetArrayFromImage(img_3)

        cs = cut_edge(img_T1)
        T1 = img_T1[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]
        T2 = img_T2[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]
        gt = img_gt[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]

        T1_padded, _ = pad_to_multiple(T1, args.patch_size)
        T2_padded, _ = pad_to_multiple(T2, args.patch_size)
        gt_padded, _ = pad_to_multiple(gt, args.patch_size)

        extract_patches_64(
            T1_padded, T2_padded, gt_padded,
            patch_size=args.patch_size,
            stride_d=args.stride_d, stride_h=args.stride_h, stride_w=args.stride_w,
            base_name=file,
            des=dir_T1_patch, des1=dir_T2_patch, des2=dir_Seg_patch, des_zero=dir_0_patch
        )

    print('All done!!!')
