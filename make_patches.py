import os
import numpy as np
import SimpleITK as sitk
import torch


def extract_ordered_patches(path, path1, path2, patch_size: tuple, stride_size: tuple, des, des1, des2, des_zero, name):
# def extract_ordered_patches(path, path2, patch_size: tuple, stride_size: tuple, des, des2, des_zero, name):
# def extract_ordered_patches(path, patch_size: tuple, stride_size: tuple, des, des_zero, name):
    File = sitk.ReadImage(path)
    spacing = File.GetSpacing()
    origin = File.GetOrigin()
    dir = File.GetDirection()
    imgs = sitk.GetArrayFromImage(File)
    File1 = sitk.ReadImage(path1)
    imgs1 = sitk.GetArrayFromImage(File1)
    File2 = sitk.ReadImage(path2)
    imgs2 = sitk.GetArrayFromImage(File2)
    assert imgs.ndim > 2
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=0)
    h, w, z = imgs.shape
    patch_h, patch_w, patch_z = patch_size
    stride_h, stride_w, stride_z = stride_size

    # assert (h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0 and (z - patch_z) % stride_z == 0
    if (h - patch_h) % stride_h == 0:
        n_patches_y = (h - patch_h) // stride_h + 1
    else:
        n_patches_y = (h - patch_h) // stride_h + 2

    if (w - patch_w) % stride_w == 0:
        n_patches_x = (w - patch_w) // stride_w + 1
    else:
        n_patches_x = (w - patch_w) // stride_w + 2

    if (z - patch_z) % stride_z == 0:
        n_patches_z = (z - patch_z) // stride_z + 1
    else:
        n_patches_z = (z - patch_z) // stride_z + 2
    n_patches_per_img = n_patches_x * n_patches_y * n_patches_z
    n_patches = n_patches_per_img
    patches = np.empty((n_patches, patch_h, patch_w, patch_z), dtype=imgs.dtype)
    patch_idx = 1
    a = 0
    # for img in imgs:
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            for k in range(n_patches_z):
                if (i * stride_h + patch_h) > h:
                    y1 = h - patch_h
                    y2 = y1 + patch_h
                else:
                    y1 = i * stride_h
                    y2 = y1 + patch_h

                if (j * stride_w + patch_w) > w:
                    x1 = w - patch_w
                    x2 = x1 + patch_w
                else:
                    x1 = j * stride_w
                    x2 = x1 + patch_w

                if (k * stride_z + patch_z) > z:
                    z1 = z - patch_z
                    z2 = z1 + patch_z
                else:
                    z1 = k * stride_z
                    z2 = z1 + patch_z
                if patch_idx < 10:
                    patch_name = name.split('.nii.gz')[0] + '_00' + str(patch_idx) + '.nii.gz'
                elif 9 < patch_idx < 100:
                    patch_name = name.split('.nii.gz')[0] + '_0' + str(patch_idx) + '.nii.gz'
                else:
                    patch_name = name.split('.nii.gz')[0] + '_' + str(patch_idx) + '.nii.gz'

                # if img_size need to be Integral multiple with stride size

                patch = sitk.GetImageFromArray(imgs[y1:y2, x1:x2, z1:z2])
                patch1 = sitk.GetImageFromArray(imgs1[y1:y2, x1:x2, z1:z2])
                patch2 = sitk.GetImageFromArray(imgs2[y1:y2, x1:x2, z1:z2])
                patch.SetSpacing(spacing)
                patch.SetOrigin(origin)
                patch.SetDirection(dir)
                patch1.SetSpacing(spacing)
                patch1.SetOrigin(origin)
                patch1.SetDirection(dir)
                patch2.SetSpacing(spacing)
                patch2.SetOrigin(origin)
                patch2.SetDirection(dir)
                if np.max(patch) != 0 and np.max(patch1) != 0 and np.max(patch2) != 0:
                # if np.max(patch) != 0 and np.max(patch2) != 0:
                # if np.max(patch) != 0:

                    sitk.WriteImage(patch, os.path.join(des, patch_name))

                    sitk.WriteImage(patch1, os.path.join(des1, patch_name.replace('T1', 'T2')))

                    sitk.WriteImage(patch2, os.path.join(des2, patch_name.replace('.nii', '_seg.nii')))
                    patch_idx += 1
                else:
                    sitk.WriteImage(patch, os.path.join(des_zero, patch_name))
                    sitk.WriteImage(patch1, os.path.join(des_zero, patch_name.replace('T1', 'T2')))
                    sitk.WriteImage(patch2, os.path.join(des_zero, patch_name.replace('.nii', '_seg.nii')))
                    patch_idx += 1
                    a += 1

                # patches[patch_idx] = img[y1:y2, x1:x2, z1:z2]

    print(str(patch_idx - 1) + "done!")
    print(a, 'patches are all 0')


def rebuild_images(patches, img_size: tuple, stride_size: tuple):
    assert patches.ndim == 4

    img_h, img_w, img_z = img_size
    stride_h, stride_w, stride_z = stride_size
    n_patches, patch_h, patch_w, patch_z = patches.shape

    assert (img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0 and (img_z - patch_z) % stride_z == 0

    n_patches_y = (img_h - patch_h) // stride_h + 1
    n_patches_x = (img_w - patch_w) // stride_w + 1
    n_patches_z = (img_z - patch_z) // stride_z + 1
    n_patches_per_img = n_patches_x * n_patches_y * n_patches_z
    print(n_patches_x)
    print(n_patches_per_img)
    batch_size = n_patches // n_patches_per_img
    imgs = np.zeros((batch_size, img_h, img_w, img_z))
    weights = np.zeros_like(imgs)
    # print(weights)

    for img_idx, (img, weights) in enumerate(zip(imgs, weights)):
        start = img_idx * n_patches_per_img

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                for k in range(n_patches_z):
                    y1 = i * stride_h
                    y2 = y1 + patch_h
                    x1 = j * stride_w
                    x2 = x1 + patch_w
                    z1 = k * stride_z
                    z2 = z1 + patch_z
                    patch_idx = start + i * n_patches_x * n_patches_z + k
                    # print(patch_idx)
                    img[y1:y2, x1:x2, z1:z2] += patches[patch_idx]
                    weights[y1:y2, x1:x2, z1:z2] += 1
                    # print(img[0])
    # print(imgs[0][0][70:100])
    imgs /= weights
    # print(imgs.astype(patches.dtype).shape)
    return imgs.astype(patches.dtype)


def prepare_patch(cutted_image, patch_size, stride_size):
    """Determine patches for validation."""

    patch_ids = []

    D, H, W, _ = cutted_image.shape

    drange = list(range(0, D - patch_size + 1, stride_size))
    hrange = list(range(0, H - patch_size + 1, stride_size))
    wrange = list(range(0, W - patch_size + 1, stride_size))

    if (D - patch_size) % stride_size != 0:
        drange.append(D - patch_size)
    if (H - patch_size) % stride_size != 0:
        hrange.append(H - patch_size)
    if (W - patch_size) % stride_size != 0:
        wrange.append(W - patch_size)

    for d in drange:
        for h in hrange:
            for w in wrange:
                patch_ids.append((d, h, w))

    return patch_ids


def patches(imgs, imgs1, imgs2, patch_size: tuple, stride_size: tuple):
    list_A = []
    list_B = []
    list_S = []
    assert imgs.ndim > 2
    # if imgs.ndim == 3:
    #     imgs = np.expand_dims(imgs, axis=0)
    h, w, z = imgs.shape
    patch_h, patch_w, patch_z = patch_size
    stride_h, stride_w, stride_z = stride_size

    # assert (h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0 and (z - patch_z) % stride_z == 0
    if (h - patch_h) % stride_h == 0:
        n_patches_y = (h - patch_h) // stride_h + 1
    else:
        n_patches_y = (h - patch_h) // stride_h + 2

    if (w - patch_w) % stride_w == 0:
        n_patches_x = (w - patch_w) // stride_w + 1
    else:
        n_patches_x = (w - patch_w) // stride_w + 2

    if (z - patch_z) % stride_z == 0:
        n_patches_z = (z - patch_z) // stride_z + 1
    else:
        n_patches_z = (z - patch_z) // stride_z + 2
    n_patches_per_img = n_patches_x * n_patches_y * n_patches_z
    n_patches = n_patches_per_img
    patches = np.empty((n_patches, patch_h, patch_w, patch_z), dtype=imgs.dtype)
    patch_idx = 1
    a = 0
    # for img in imgs:
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            for k in range(n_patches_z):
                if (i * stride_h + patch_h) > h:
                    y1 = h - patch_h
                    y2 = y1 + patch_h
                else:
                    y1 = i * stride_h
                    y2 = y1 + patch_h

                if (j * stride_w + patch_w) > w:
                    x1 = w - patch_w
                    x2 = x1 + patch_w
                else:
                    x1 = j * stride_w
                    x2 = x1 + patch_w

                if (k * stride_z + patch_z) > z:
                    z1 = z - patch_z
                    z2 = z1 + patch_z
                else:
                    z1 = k * stride_z
                    z2 = z1 + patch_z

                # if img_size need to be Integral multiple with stride size

                patch = sitk.GetImageFromArray(imgs[y1:y2, x1:x2, z1:z2])
                patch1 = sitk.GetImageFromArray(imgs1[y1:y2, x1:x2, z1:z2])
                patch2 = sitk.GetImageFromArray(imgs2[y1:y2, x1:x2, z1:z2])
                if np.max(patch) != 0 and np.max(patch1) != 0 and np.max(patch2) != 0:
                    list_A.append(imgs[y1:y2, x1:x2, z1:z2])
                    list_B.append(imgs1[y1:y2, x1:x2, z1:z2])
                    list_S.append(imgs2[y1:y2, x1:x2, z1:z2])

                    # patch_idx += 1

    return list_A, list_B, list_S
    # patches[patch_idx] = img[y1:y2, x1:x2, z1:z2]

    # print(str(patch_idx - 1) + "done!")
    # print(a, 'patches are all 0')


if __name__ == '__main__':
    T1_path = None                          #str './'
    T2_path = None
    Seg_path = None
    dir_T1 = None
    dir_T2 = None
    dir_Seg = None
    dir_0 = None
    assert T1_path is not None, "file_path_T1 CANNOT BE NONE"
    assert T2_path is not None, "file_path_T2 CANNOT BE NONE"
    assert Seg_path is not None, "file_dir_gt CANNOT BE NONE"
    assert dir_T1 is not None, "file_dir_T1 CANNOT BE NONE"
    assert dir_T2 is not None, "file_dir_T2 CANNOT BE NONE"
    assert dir_Seg is not None, "file_dir_gt CANNOT BE NONE"
    assert dir_0 is not None, "file_dir_0 CANNOT BE NONE"

    list_T1 = [i for i in os.listdir(T1_path) if i.endswith('.nii.gz')]


    for i in sorted(list_T1):
        print(i)
        path = os.path.join(T1_path, i)
        path1 = os.path.join(T2_path, i.replace('T1', 'T2'))
        path2 = os.path.join(Seg_path, i.replace('.nii', '_seg.nii'))

        extract_ordered_patches(path, path1, path2, (64, 64, 64), (32, 32, 32), dir_T1, dir_T2, dir_Seg, dir_0, i)
        # extract_ordered_patches(path, path2, (64, 64, 64), (32, 32, 32), dir_T1, dir_Seg, dir_0, i)
        # extract_ordered_patches(path, (64, 64, 64), (32, 32, 32), dir_T1, dir_0, i)  # only T1

    print(' done !!!')
