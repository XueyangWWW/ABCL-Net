import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
from data.image_folder import make_dataset, norm_img, get_bounds
import numpy as np
import SimpleITK as sitk


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


class TrainDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot_train

        self.dir_A = os.path.join(self.root, 'T1')
        self.dir_B = os.path.join(self.root, 'T2')
        self.dir_Seg = os.path.join(self.root, 'label')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.Seg_paths = sorted(make_dataset(self.dir_Seg))
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)


    def __getitem__(self, index):

        A_path = os.path.join(self.dir_A, self.A_paths[index])
        Seg_path = os.path.join(self.dir_Seg, self.Seg_paths[index])

        B_path = os.path.join(self.dir_B, self.B_paths[index])

        A_img = sitk.ReadImage(A_path)
        A_img = sitk.GetArrayFromImage(A_img)
        B_img = sitk.ReadImage(B_path)
        B_img = sitk.GetArrayFromImage(B_img)
        Seg_img = sitk.ReadImage(Seg_path)
        Seg_img = sitk.GetArrayFromImage(Seg_img)


        A_tensor = torch.from_numpy(A_img).to(dtype=torch.float)
        B_tensor = torch.from_numpy(B_img).to(dtype=torch.float)
        Seg_tensor_ori = torch.from_numpy(Seg_img).to(dtype=torch.float)
        p1 = random.randint(0, 1)
        p2 = random.randint(0, 1)
        p3 = random.randint(30, 60)

        aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p1),  # 随机水平翻转
            transforms.RandomVerticalFlip(p2),  # 随机垂直翻转
            transforms.RandomRotation(degrees=(p3, p3))   # 随机旋转，你可以根据需要调整角度
            # transforms.RandomResizedCrop(64, scale=(0.8, 1.0))  # 随机裁剪并重新调整大小
        ])

        random_number = random.random()
        if random_number < 0.8:
            A_tensor = aug(A_tensor)
            B_tensor = aug(B_tensor)
            Seg_tensor_ori = aug(Seg_tensor_ori)


        if torch.max(A_tensor) == torch.min(A_tensor):
            A_tensor = A_tensor
        else:
            A_tensor = (A_tensor - torch.min(A_tensor)) / (torch.max(A_tensor) - torch.min(A_tensor))
            A_tensor = 2 * A_tensor - 1

        if torch.max(B_tensor) == torch.min(B_tensor):
            B_tensor = B_tensor
        else:
            B_tensor = (B_tensor - torch.min(B_tensor)) / (torch.max(B_tensor) - torch.min(B_tensor))
            B_tensor = 2 * B_tensor - 1
        Seg_tensor = to_categorical(Seg_tensor_ori, 4)

        #         target = target.astype(np.float32)

        return {'img_A': A_tensor, 'img_B': B_tensor, 'Seg': Seg_tensor,
                'Seg_ori': Seg_tensor_ori, 'A_paths': A_path}



    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'TrainDataset'



class ValDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot_val

        self.dir_A = os.path.join(self.root, 'T1')
        self.dir_B = os.path.join(self.root, 'T2')
        self.dir_Seg = os.path.join(self.root, 'label')

        self.A_filenames = sorted(make_dataset(self.dir_A))
        self.B_filenames = sorted(make_dataset(self.dir_B))
        self.seg_filenames = sorted(make_dataset(self.dir_Seg))

        self.A_size = len(self.A_filenames)

    def __getitem__(self, index):
        A_filename = self.A_filenames[index]
        A_path = os.path.join(self.dir_A, A_filename)
        A_img = sitk.ReadImage(A_path)
        A_img = sitk.GetArrayFromImage(A_img)

        B_filename = self.B_filenames[index]
        B_path = os.path.join(self.dir_B, B_filename)
        B_img = sitk.ReadImage(B_path)
        B_img = sitk.GetArrayFromImage(B_img)

        if np.max(A_img) == np.min(A_img):
            A_img = A_img
        else:
            A_img = (A_img - np.min(A_img)) / (np.max(A_img) - np.min(A_img))
            A_img = 2 * A_img - 1
        if np.max(B_img) == np.min(B_img):
            B_img = B_img
        else:
            B_img = (B_img - np.min(B_img)) / (np.max(B_img) - np.min(B_img))
            B_img = 2 * B_img - 1
        A_tensor = torch.from_numpy(A_img).to(dtype=torch.float)
        B_tensor = torch.from_numpy(B_img).to(dtype=torch.float)


        Seg_filename = self.seg_filenames[index]
        Seg_path = os.path.join(self.dir_Seg, Seg_filename)

        Seg_img = sitk.ReadImage(Seg_path)
        Seg_img = sitk.GetArrayFromImage(Seg_img)
        Seg_ori = torch.from_numpy(Seg_img).to(dtype=torch.float)
        Seg_tensor = to_categorical(Seg_ori, 4)
        return {'img_A': A_tensor, 'Seg': Seg_tensor, 'img_B': B_tensor, 'A_paths': A_path, 'B_paths': B_path, 'Seg_paths': Seg_path}

    def __len__(self):

        return self.A_size

    def name(self):
        return 'ValDataset'



class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.dir_A = opt.test_A_dir
        # self.dir_B = opt.test_B_dir
        # self.dir_seg = opt.test_seg_dir


        self.A_filenames = sorted(make_dataset(self.dir_A))
        # self.B_filenames = sorted(make_dataset(self.dir_B))
        # self.seg_filenames = sorted(make_dataset(self.dir_seg))
        self.A_size = len(self.A_filenames)



    def __getitem__(self, index):
        A_filename = self.A_filenames[index]
        A_path = os.path.join(self.dir_A, A_filename)
        A_img = sitk.ReadImage(A_path)
        A_img = sitk.GetArrayFromImage(A_img)

        # B_filename = self.B_filenames[index]
        # B_path = os.path.join(self.dir_B, B_filename)
        # B_img = sitk.ReadImage(B_path)
        # B_img = sitk.GetArrayFromImage(B_img)
        if np.max(A_img) == np.min(A_img):
            A_img = A_img
        else:
            A_img = (A_img - np.min(A_img)) / (np.max(A_img) - np.min(A_img))
            A_img = 2 * A_img - 1
        # if np.max(B_img) == np.min(B_img):
        #     B_img = B_img
        # else:
        #     B_img = (B_img - np.min(B_img)) / (np.max(B_img) - np.min(B_img))
        #     B_img = 2 * B_img - 1

        A_tensor = torch.from_numpy(A_img).to(dtype=torch.float)
        # B_tensor = torch.from_numpy(B_img).to(dtype=torch.float)


        # Seg_filename = self.seg_filenames[index]
        # Seg_path = os.path.join(self.dir_seg, Seg_filename)
        #
        # Seg_img = sitk.ReadImage(Seg_path)
        # Seg_img = sitk.GetArrayFromImage(Seg_img)
        # Seg_tensor = torch.from_numpy(Seg_img).to(dtype=torch.float)





        # Seg_img = self.transforms_normalize(Seg_img)
        # Seg_img[Seg_img > 0] = 1
        #
        # Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.crop_size, self.opt.crop_size)
        # Seg_imgs[0, :, :] = Seg_img == 1

        # return {'A_img': A_tensor, 'Seg_img': Seg_tensor, 'B_img': B_tensor,
        #         'A_paths': A_path, 'Seg_paths': Seg_path}
        return {'A_img': A_tensor, 'A_paths': A_path}  # UWM_590

    def __len__(self):
        return self.A_size

    #
    def name(self):
        return 'TestDataset'
