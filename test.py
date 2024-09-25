import shutil
import time
import os

import numpy as np
import torch.nn as nn
import SimpleITK as sitk
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from patch import extract_ordered_patches, rebuild_images
from metrics_seg import seg_metric_test, gen_metric_test
from crop import crop_syn_t2


if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    seg_output_dir = opt.test_seg_output_dir


    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_test_data()
    dataset_size = len(dataset)
    print('#test images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)

    save_dir_seg = os.path.join(seg_output_dir, 'patches_seg')
    save_dir_fakeT2 = os.path.join(seg_output_dir, 'patches_syn_t2')
    save_dir_img_seg = os.path.join(seg_output_dir, 'imgs_seg')
    save_dir_img_fakeT2 = os.path.join(seg_output_dir, 'imgs_syn_t2')
    # save_dir_img_fakeT2_crop = os.path.join(seg_output_dir, 'imgs_syn_t2_crop')

    ori_T1_path = './'  # ORI T1 images, size of 160x160x160
    ori_T2_path = './' # ORI T2 images, size of 160x160x160
    # ori_T2_path_crop = '/home/ubuntu/wuxueyang/AccsegNet_3D/dataroot/data/test/crop_160/T2_crop/'
    ori_seg_path = './' # ORI label images, size of 160x160x160
    file_example = os.listdir(ori_T1_path)[0]
    file = sitk.ReadImage(os.path.join(ori_T1_path, file_example))
    ori = file.GetOrigin()
    spa = file.GetSpacing()
    dir = file.GetDirection()

    path_test_all0 = opt.test_0_dir
    if not os.path.exists(save_dir_seg):
        os.makedirs(save_dir_seg)
    if not os.path.exists(save_dir_fakeT2):
        os.makedirs(save_dir_fakeT2)
    if not os.path.exists(save_dir_img_seg):
        os.makedirs(save_dir_img_seg)
    if not os.path.exists(save_dir_img_fakeT2):
        os.makedirs(save_dir_img_fakeT2)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()



        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        # print('processing image... %s' % img_path)

        visualizer.save_seg_images_to_dir(save_dir_seg, save_dir_fakeT2, visuals, img_path)
        i += 1
        print(i, ' test done')
    list_0 = os.listdir(path_test_all0)
    for file in list_0:

        if file[13:16] == 'Seg':
            old_path = os.path.join(path_test_all0, file)
            new_path = os.path.join(save_dir_seg, file)
            shutil.copyfile(old_path, new_path)
        if file[13:15] == 'T2':
            old_path = os.path.join(path_test_all0, file)
            new_path = os.path.join(save_dir_fakeT2, file)
            shutil.copyfile(old_path, new_path)

    print('prepare to rebuild the %s images...' % len(os.listdir(save_dir_seg)))
    patch_list_seg = sorted(os.listdir(save_dir_seg))
    patch_list_ft2 = sorted(os.listdir(save_dir_fakeT2))
    patches_seg = np.zeros((64, 64, 64, 64))
    patches_ft2 = np.zeros((64, 64, 64, 64))
    a = 0
    b = 1
    d = 0
    for patch in patch_list_seg:
        print(patch)
        file = os.path.join(save_dir_seg, patch)

        img = sitk.ReadImage(file)

        img = sitk.GetArrayFromImage(img)
        img = np.expand_dims(img, axis=0)
        patches_seg[a, 0:64, 0:64, 0:64] = img
        a += 1
        if a == 64:
            raw_image = rebuild_images(patches_seg, (160, 160, 160), (32, 32, 32))
            raw_image = np.round(raw_image)
            raw_image = np.squeeze(raw_image, axis=0)
            patch_idx = sitk.GetImageFromArray(raw_image)
            patch_idx.SetOrigin(ori)
            patch_idx.SetSpacing(spa)
            patch_idx.SetDirection(dir)
            sitk.WriteImage(patch_idx, save_dir_img_seg + '/' + patch[:13] + 'Seg.nii.gz')
            patches = np.zeros((64, 64, 64, 64))
            a = 0
    for patch in patch_list_ft2:
        file = os.path.join(save_dir_fakeT2, patch)
        img = sitk.ReadImage(file)
        img = sitk.GetArrayFromImage(img)
        img = np.expand_dims(img, axis=0)
        patches_ft2[d, 0:64, 0:64, 0:64] = img
        d += 1
        if d == 64:
            raw_image = rebuild_images(patches_ft2, (160, 160, 160), (32, 32, 32))
            raw_image = np.round(raw_image)
            raw_image = np.squeeze(raw_image, axis=0)
            patch_idx = sitk.GetImageFromArray(raw_image)
            patch_idx.SetOrigin(ori)
            patch_idx.SetSpacing(spa)
            patch_idx.SetDirection(dir)
            sitk.WriteImage(patch_idx, save_dir_img_fakeT2 + '/' + patch[:13] + 'T2.nii.gz')
            patches = np.zeros((64, 64, 64, 64))
            d = 0
            print(str(b) + " images have been rebuilt!")
            b += 1
    print("All images done!")

    # crop_syn_t2(save_dir_img_fakeT2, ori_T2_path, save_dir_img_fakeT2_crop, ori_T2_path_crop)

    seg_metric_test(save_dir_img_seg, ori_seg_path, seg_output_dir)
    # gen_metric_test(save_dir_img_fakeT2_crop, ori_T2_path_crop, seg_output_dir)