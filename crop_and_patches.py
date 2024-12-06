import os
import SimpleITK as sitk
import numpy as np

##################################
# Part 1: Cropping
##################################

def cut_edge(data):
    """
    Automatically trim zero-valued edges from a 3D image array.
    This finds the bounding box that includes all non-zero voxels.

    Args:
        data (ndarray): 3D numpy array [D, H, W].
    Returns:
        original_shape (list): The original shape [D, H, W].
        cut_size (list): [D_start, D_end, H_start, H_end, W_start, W_end]
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

    original_shape = [D, H, W]
    cut_size = [int(D_s), int(D_e + 1), int(H_s), int(H_e + 1), int(W_s), int(W_e + 1)]
    return (original_shape, cut_size)


def fixed_crop(data, img_size):
    """
    Crop the 3D image to a fixed cubic size (img_size x img_size x img_size).
    The image is cropped centrally.
    """
    D, H, W = data.shape
    a = (D - img_size) / 2
    b = (H - img_size) / 2
    c = (W - img_size) / 2
    cut_size = [int(a), int(D - a), int(b), int(H - b), int(c), int(W - c)]
    return cut_size

def crop_images(file_path_T1, file_path_T2, file_path_gt, file_dir_T1, file_dir_T2, file_dir_gt, crop_size=160):
    """
    Crop original T1, T2, and Seg images to a fixed size and save them.
    """
    os.makedirs(file_dir_T1, exist_ok=True)
    os.makedirs(file_dir_T2, exist_ok=True)
    os.makedirs(file_dir_gt, exist_ok=True)

    file_T1 = sorted([f for f in os.listdir(file_path_T1) if f.endswith('.nii.gz')])
    for file in file_T1:
        fname_T1 = os.path.join(file_path_T1, file)
        fname_T2 = os.path.join(file_path_T2, file.replace('T1', 'T2'))
        fname_gt = os.path.join(file_path_gt, file.replace('T1', 'Seg'))

        img_1 = sitk.ReadImage(fname_T1)
        spacing = img_1.GetSpacing()
        origin = img_1.GetOrigin()
        direction = img_1.GetDirection()
        img_T1 = sitk.GetArrayFromImage(img_1)

        img_2 = sitk.ReadImage(fname_T2)
        img_T2 = sitk.GetArrayFromImage(img_2)

        img_3 = sitk.ReadImage(fname_gt)
        img_gt = sitk.GetArrayFromImage(img_3)

        # Crop to fixed size
        cs = fixed_crop(img_T1, crop_size)
        T1 = img_T1[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]
        T2 = img_T2[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]
        gt = img_gt[cs[0]:cs[1], cs[2]:cs[3], cs[4]:cs[5]]

        # Convert back to ITK images and keep metadata
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

        # Save cropped images
        sitk.WriteImage(img_t1, os.path.join(file_dir_T1, file))
        sitk.WriteImage(img_t2, os.path.join(file_dir_T2, file.replace('T1', 'T2')))
        sitk.WriteImage(img_GT, os.path.join(file_dir_gt, file.replace('T1', 'Seg')))

    print("Cropping done!")


##################################
# Part 2: Patch Extraction
##################################

def extract_ordered_patches(path, path1, path2, patch_size: tuple, stride_size: tuple, des, des1, des2, des_zero, name):
    """
    Extract ordered 3D patches from cropped T1, T2, and Seg images and save them.
    """
    File = sitk.ReadImage(path)
    spacing = File.GetSpacing()
    origin = File.GetOrigin()
    dir = File.GetDirection()
    imgs = sitk.GetArrayFromImage(File)

    File1 = sitk.ReadImage(path1)
    imgs1 = sitk.GetArrayFromImage(File1)

    File2 = sitk.ReadImage(path2)
    imgs2 = sitk.GetArrayFromImage(File2)

    h, w, z = imgs.shape
    patch_h, patch_w, patch_z = patch_size
    stride_h, stride_w, stride_z = stride_size

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
    patch_idx = 1
    count_zero = 0

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
                elif patch_idx < 100:
                    patch_name = name.split('.nii.gz')[0] + '_0' + str(patch_idx) + '.nii.gz'
                else:
                    patch_name = name.split('.nii.gz')[0] + '_' + str(patch_idx) + '.nii.gz'

                patch_arr = imgs[y1:y2, x1:x2, z1:z2]
                patch_arr1 = imgs1[y1:y2, x1:x2, z1:z2]
                patch_arr2 = imgs2[y1:y2, x1:x2, z1:z2]

                patch = sitk.GetImageFromArray(patch_arr)
                patch1 = sitk.GetImageFromArray(patch_arr1)
                patch2 = sitk.GetImageFromArray(patch_arr2)

                patch.SetSpacing(spacing)
                patch.SetOrigin(origin)
                patch.SetDirection(dir)

                patch1.SetSpacing(spacing)
                patch1.SetOrigin(origin)
                patch1.SetDirection(dir)

                patch2.SetSpacing(spacing)
                patch2.SetOrigin(origin)
                patch2.SetDirection(dir)

                # Save patches
                if np.max(patch_arr) != 0 and np.max(patch_arr1) != 0 and np.max(patch_arr2) != 0:
                    # non-zero patch
                    sitk.WriteImage(patch, os.path.join(des, patch_name))
                    sitk.WriteImage(patch1, os.path.join(des1, patch_name.replace('T1', 'T2')))
                    sitk.WriteImage(patch2, os.path.join(des2, patch_name.replace('.nii', '_seg.nii')))
                else:
                    # zero patch
                    sitk.WriteImage(patch, os.path.join(des_zero, patch_name))
                    sitk.WriteImage(patch1, os.path.join(des_zero, patch_name.replace('T1', 'T2')))
                    sitk.WriteImage(patch2, os.path.join(des_zero, patch_name.replace('.nii', '_seg.nii')))
                    count_zero += 1
                patch_idx += 1

    print(str(patch_idx - 1) + " patches extracted!")
    print(count_zero, 'patches are all zero')


if __name__ == '__main__':
    #====================
    # Step 1: Cropping
    #====================
    base_dir = 'MRI_data/'
    file_path_T1 = os.path.join(base_dir, 'T1w')
    file_path_T2 = os.path.join(base_dir, 'T2w')
    file_path_gt = os.path.join(base_dir, 'Tissue')

    out_base_dir = 'crop_MRI_data/'
    file_dir_T1 = os.path.join(out_base_dir, 'T1w')
    file_dir_T2 = os.path.join(out_base_dir, 'T2w')
    file_dir_gt = os.path.join(out_base_dir, 'Tissue')

    # Make sure these directories exist
    os.makedirs(file_dir_T1, exist_ok=True)
    os.makedirs(file_dir_T2, exist_ok=True)
    os.makedirs(file_dir_gt, exist_ok=True)

    crop_images(file_path_T1, file_path_T2, file_path_gt, file_dir_T1, file_dir_T2, file_dir_gt, crop_size=160)

    #====================
    # Step 2: Patch Extraction
    #====================

    # Paths to the cropped images
    T1_path = file_dir_T1
    T2_path = file_dir_T2
    Seg_path = file_dir_gt

    # Output directories for patches
    # Adjust these as needed
    patch_out_base = 'patch_data'
    dir_T1_patch = os.path.join(patch_out_base, 'T1')
    dir_T2_patch = os.path.join(patch_out_base, 'T2')
    dir_Seg_patch = os.path.join(patch_out_base, 'Seg')
    dir_0_patch = os.path.join(patch_out_base, 'Zero')

    os.makedirs(dir_T1_patch, exist_ok=True)
    os.makedirs(dir_T2_patch, exist_ok=True)
    os.makedirs(dir_Seg_patch, exist_ok=True)
    os.makedirs(dir_0_patch, exist_ok=True)

    # Patch parameters
    patch_size = (64, 64, 64)
    stride_size = (32, 32, 32)

    list_T1_imgs = [i for i in os.listdir(T1_path) if i.endswith('.nii.gz')]
    for i in sorted(list_T1_imgs):
        print("Processing:", i)
        path = os.path.join(T1_path, i)
        path1 = os.path.join(T2_path, i.replace('T1', 'T2'))
        path2 = os.path.join(Seg_path, i.replace('.nii', '_seg.nii'))

        extract_ordered_patches(path, path1, path2, patch_size, stride_size, 
                                dir_T1_patch, dir_T2_patch, dir_Seg_patch, dir_0_patch, i)

    print('Patch extraction done!!!')
