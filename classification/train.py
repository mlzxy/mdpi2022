# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


def fix_shapes(model, state_dict):
    new_state_dict = model.state_dict()
    for k,v in state_dict.items():
        if v.shape != new_state_dict[k].shape:
            print(f"{k} shape mismatch: {v.shape} vs {new_state_dict[k].shape}")
            state_dict[k] = new_state_dict[k]
            continue


def train(epoch):

    start = time.time()
    # net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        # import ipdb; ipdb.set_trace()
        loss = loss_function(outputs, labels)
        loss.backward()
        if net.training:
            optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg', help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--resume_from', default="", help="folder to load resume checkpoint")
    parser.add_argument('--resume_after_convert', default=False)
    parser.add_argument("--no_save", default=False, action='store_true')
    parser.add_argument("--use_cifar10", default=False, action='store_true')
    args = parser.parse_args()
    
    ##############################################################################
    from exp_helper import init
    init([settings, args])
    ##############################################################################


    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        shrink=args.train_set_shrink if hasattr(args, "train_set_shrink") else None,
        use_cifar10=args.use_cifar10
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        use_cifar10=args.use_cifar10
    )

    ##############################################################################
    from exp_helper import set_epoch_size
    set_epoch_size(cifar100_training_loader)
    ##############################################################################


    net = get_network(args)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    def load_pretrain():
        resume_from = args.resume_from or settings.CHECKPOINT_PATH
        recent_folder = most_recent_folder(os.path.join(args.resume_from, args.net), fmt=settings.DATE_FORMAT)
        recent_weights_file = most_recent_weights(os.path.join(resume_from, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(resume_from, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        state_dict = torch.load(weights_path)
        for k in list(state_dict.keys()):
            if k.endswith('magnitude'): # only need the mask!
                state_dict.pop(k)
        fix_shapes(net, state_dict) # test _t
        net.load_state_dict(state_dict)

        for n in net.modules():
            if n.__class__.__name__ == "PruneLayer":
                if n.callback.preserve_existing_mask:
                    n._n_updates.data[:] = 0


    if args.resume and not args.resume_after_convert:
        load_pretrain()

    ##############################################################################
    from exp_helper import convert, epoch_callback

    if not args.resume_after_convert:
        net = convert(net)
    else:
        net = convert(net)
        net(torch.rand(args.b, 3, 32, 32).cuda())
        load_pretrain()
    ##############################################################################

        

    acc_valid_records = []

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        net = epoch_callback(net, epoch - 1)

        train(epoch)

        if hasattr(args, "eval_every"):
            acc = -1
            if epoch % args.eval_every == 0:
                acc = eval_training(epoch)
        else:
            acc = eval_training(epoch)

        if hasattr(settings, "record_after"):
            if epoch >= settings.record_after:
                acc_valid_records.append(float(acc))

        #start to save best performance model after learning rate decay to 0.01
        if (len(settings.MILESTONES) <= 1 or epoch > settings.MILESTONES[1]) and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            if not args.no_save:
                torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            if not args.no_save:
                torch.save(net.state_dict(), weights_path)

    writer.close()

    acc_valid_records = sorted(acc_valid_records, reverse=True)[:1]
    print(acc_valid_records)