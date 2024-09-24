import time
import os

import torch
from options.train_options import TrainOptions
import random
from data.data_loader import CreateDataLoader
from models import create_model

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.visualizer import Visualizer

if __name__ == '__main__':

    opt = TrainOptions().parse()


    data_loader = CreateDataLoader(opt)
    train_dataset, val_dataset = data_loader.load_train_data()
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    print('#training images = %d, validating images = %d' % (train_dataset_size * opt.batch_size, val_dataset_size * opt.batch_size))
    print('batch_size = %d' % opt.batch_size)
    model = create_model(opt)

    visualizer = Visualizer(opt)
    opt.visualizer = visualizer
    total_iters = 0
    print('#model created')

    loss_logs_path = os.path.join(opt.checkpoints_dir, opt.name)
    writer = SummaryWriter(loss_logs_path)

    optimize_time = 0.1

    times = []
    n = 1
    Dice = []
    # SSIM = []
    PSNR = []
    max_dice = 0
    max_psnr = 0
    best_dice_epoch = 0
    best_psnr_epoch = 0
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()

            model.set_input(data)
            model.optimize_parameters()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / opt.batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % 100 == 0:
                losses, metrics = model.get_current_losses()
                for k, v in losses.items():
                    writer.add_scalar("train_loss_%s" % k, v, total_iters)
                visualizer.print_current_losses(epoch, epoch_iter, losses, metrics, optimize_time)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     print(opt.name)  # it's useful to occasionally show the experiment name on console
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        total_test_dice = []
        total_test_psnr = []
        with torch.no_grad():
            for data in val_dataset:
                dice, psnr = model.start_validating(data)
                total_test_dice.append(dice)
                total_test_psnr.append(psnr)

        writer.add_scalar("val_dice", np.mean(total_test_dice), epoch)
        writer.add_scalar("val_psnr", np.mean(total_test_psnr), epoch)
        Dice.append(np.mean(total_test_dice))
        PSNR.append(np.mean(total_test_psnr))
        print('VAL_DICE:  ', Dice)
        print('VAL_PSNR:  ', PSNR)

        if Dice[epoch - opt.epoch_count] > max_dice:
            model.save_networks('Best_dice')
            print('%d model becomes current --Best_dice-- Model ! Congratulations !' % epoch)
            best_dice_epoch = epoch
            max_dice = Dice[epoch - opt.epoch_count]

        if PSNR[epoch - opt.epoch_count] > max_psnr:
            model.save_networks('Best_psnr')
            print('%d model becomes current --Best_psnr-- Model ! Congratulations !' % epoch)
            best_psnr_epoch = epoch
            max_psnr = PSNR[epoch - opt.epoch_count]

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))

            model.save_networks('latest')
            # if epoch % 10 == 0:
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec \t Best Dice: %d \t Best Psnr: %d' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time, best_dice_epoch, best_psnr_epoch))
        model.update_learning_rate()

    writer.close()
