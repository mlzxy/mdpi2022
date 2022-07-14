"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import os
import os.path as osp
from options.train_options import TrainOptions
from data import create_dataset
from data.unaligned_dataset import UnalignedDataset
from data.aligned_dataset import AlignedDataset
from models import create_model
from util.visualizer import Visualizer
from pytorch_gan_metrics import get_inception_score_and_fid
import torch
import torch.nn as nn
from argparse import Namespace

from qsparse.util import logging


def create_test_loader(parent):
    opt = Namespace()
    opt.dataroot = parent.dataroot
    opt.direction = 'BtoA'
    opt.max_dataset_size = float('inf')
    opt.output_nc = 3
    opt.input_nc = 3
    opt.preprocess = 'resize_and_crop'
    opt.load_size = parent.load_size
    opt.crop_size = parent.crop_size
    opt.no_flip = False
    opt.serial_batches = False
    if parent.dataset_mode == 'unaligned':
        opt.phase = "test"
        D = UnalignedDataset(opt)
    else:
        if "facades" in opt.dataroot:
            opt.phase = "valtest"
        else:
            opt.phase = "val"
        D = AlignedDataset(opt)
    loader = torch.utils.data.DataLoader(
        D,
        batch_size=1, shuffle=False,
        num_workers=4)
    return loader


if __name__ == '__main__':
    os.chdir(osp.join(osp.dirname(__file__), ".."))
    _ = TrainOptions()   # get training options
    opt = _.parse()
    # ================================================
    from exp_helper import init
    init([opt, ])
    _.print_options(opt)
    # checkpoints_dir
    # ================================================

    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    test_dataset = create_test_loader(opt)
    net_name = 'netG' if opt.model == 'pix2pix' else 'netG_A'

    ##############################################################################
    from exp_helper import set_epoch_size
    set_epoch_size(dataset.dataloader)
    ##############################################################################

    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    total_iters = 0                # the total number of training iterations

    acc_valid_records = []




    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # if opt.model == 'pix2pix' and opt.resume_from:
        #     if epoch < 10:
        #         print("deactivate dropout in pix2pix")
        #         for name, mod in model.netG.named_modules():
        #             if isinstance(mod, nn.Dropout):
        #                 mod.eval()
        #     else:
        #         print("reactivate dropout in pix2pix")
        #         for name, mod in model.netG.named_modules():
        #             if isinstance(mod, nn.Dropout):
        #                 mod.train()

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()
        # update learning rates in the beginning of every epoch.
        model.update_learning_rate()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(
                #         epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     if not opt.no_save:
            #         print('saving the latest model (epoch %d, total_iters %d)' % (
            #             epoch, total_iters))
            #         save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #         model.save_networks(save_suffix)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            if not opt.no_save:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_iters))
                model.save_networks('latest')
                # model.save_networks(epoch)

        # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
        #     save_result = total_iters % opt.update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, True)

        if epoch % 5 == 0 or epoch == 1 or epoch > 27:
            eval_start_time = time.time()
            model.eval()
            imgs = []
            with torch.no_grad():
                for batch in test_dataset:
                    img = getattr(model, net_name)(batch['B'].cuda()).cpu()  # since B to A
                    imgs.append((img + 1) / 2)
            imgs = torch.cat(imgs, dim=0)
            IS, FID = get_inception_score_and_fid(
                imgs, opt.fid_cache, verbose=False, use_torch=False)
            if epoch >= opt.record_after:
                acc_valid_records.append(FID)
            model.train()
            eval_elapse = time.time() - eval_start_time
            logging.warning(f'Epoch - {epoch}, FID {FID}, Eval Time: {eval_elapse:.3f}s')
            # if len(acc_valid_records) > 0:
            #     logging.danger(f'Best FID {min(acc_valid_records)}')

        logging.warning('End of epoch %d / %d \t Time Taken: %d sec' % (epoch,
              opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    acc_valid_records = sorted(acc_valid_records, reverse=False)[:1]
    print(acc_valid_records)
