import numpy as np
import SimpleITK as sitk
import os
from models import create_model
from options.test_options import TestOptions
import torch

def extract_ordered_patches(imgs, patch_size: tuple, stride_size: tuple):
    assert imgs.ndim > 2
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=0)

    b, h, w, z = imgs.shape
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
    n_patches = n_patches_per_img * b
    patches = np.empty((n_patches, patch_h, patch_w, patch_z), dtype=imgs.dtype)
    patch_idx = 0
    for l in range(b):
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
                    patches[patch_idx] = imgs[l, y1:y2, x1:x2, z1:z2]
                    patch_idx += 1
    print(patch_idx)
    return patches, patch_idx


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
    # print(n_patches_x)
    # print(n_patches_per_img)
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
                    patch_idx = start + i * n_patches_y*n_patches_x +j*n_patches_z + k
                    # print(patch_idx)
                    img[y1:y2, x1:x2, z1:z2] += patches[patch_idx]
                    weights[y1:y2, x1:x2, z1:z2] += 1
                    # print(img[0])
    # print(imgs[0][0][70:100])
    imgs /= weights

    # print(imgs.astype(patches.dtype).shape)
    return imgs

