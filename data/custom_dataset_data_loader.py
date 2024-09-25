import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset_Train(opt):

    from data.gen2seg_dataset import TrainDataset, ValDataset

    train_dataset = TrainDataset()
    val_dataset = ValDataset()
    train_dataset.initialize(opt)
    val_dataset.initialize(opt)
    return train_dataset, val_dataset


def CreateDataset_Test(opt):

    from data.gen2seg_dataset import TestDataset
    test_dataset = TestDataset()

    test_dataset.initialize(opt)

    return test_dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        if opt.dataset_mode == 'gen2seg_train':
            self.train_dataset, self.val_dataset = CreateDataset_Train(opt)

            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                # shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))

            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                # shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))


        elif opt.dataset_mode == 'gen2seg_test':
            self.test_dataset = CreateDataset_Test(opt)

            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                # shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))

            self.reval_dataloader = torch.utils.data.DataLoader(
                self.reval_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                # shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))

    def load_train_data(self):
        return self.train_dataloader, self.val_dataloader

    def load_retrain_data(self):
        return self.reval_dataloader, self.retrain_dataloader

    def load_test_data(self):
        return self.test_dataloader
