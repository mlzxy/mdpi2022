import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

def fix_shapes(model, state_dict):
    new_state_dict = model.state_dict()
    for k,v in state_dict.items():
        if v.shape != new_state_dict[k].shape:
            print(f"{k} shape mismatch: {v.shape} vs {new_state_dict[k].shape}")
            state_dict[k] = new_state_dict[k]
            continue


def train(cfg, writer, logger):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    ##############################################################################
    from exp_helper import set_epoch_size
    set_epoch_size(1)
    ##############################################################################


    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)

    def load_pretrain():
        if cfg["training"]["resume"] is not None:
            if os.path.isfile(cfg["training"]["resume"]):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
                )
                checkpoint = torch.load(cfg["training"]["resume"])
                for k in list(checkpoint["model_state"].keys()):
                    if k.endswith('magnitude'): # only need the mask!
                        checkpoint["model_state"].pop(k)
                
                fix_shapes(model, checkpoint["model_state"])
                model.load_state_dict(checkpoint["model_state"])
                logger.info(
                    "Loaded checkpoint '{}' (iter {})".format(
                        cfg["training"]["resume"], checkpoint["epoch"]
                    )
                )
            else:
                logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))
    
    if "resume_after_convert" not in cfg["training"] or not cfg["training"]["resume_after_convert"]:
        load_pretrain()

    ##############################################################################
    from exp_helper import convert
    if "resume_after_convert" in cfg["training"] and cfg["training"]["resume_after_convert"]:
        model = convert(model)
        model(torch.rand(cfg["training"]["batch_size"], 3, cfg["data"]["img_rows"], cfg["data"]["img_cols"]).cuda())
        load_pretrain()
        # model = convert(model, "quantize")
    else:
        model = convert(model)
    ##############################################################################

    # if "resume_after_convert" in cfg["training"] and cfg["training"]["resume_after_convert"]:
    #     print("loading after-convert checkpoint")
    #     model(torch.rand(cfg["training"]["batch_size"], 3, cfg["data"]["img_rows"], cfg["data"]["img_cols"]).cuda())
    #     load_pretrain()

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    ##############################################################################
    from exp_helper import epoch_callback
    ##############################################################################
    acc_valid_records = []

    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels) in trainloader:
            ##############################################################################
            from exp_helper import epoch_callback
            model = epoch_callback(model, i)
            # model.train()
            ##############################################################################

            i += 1
            start_ts = time.time()
            scheduler.step()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            if model.training:
                optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time.time() - start_ts,
                )

                # print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in enumerate(valloader):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    # print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if hasattr(cfg, "record_after"):
                    if (i+1) >= cfg.record_after:
                        acc_valid_records.append(score["Mean IoU : \t"])

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    if not cfg.get("no_save", False):
                        torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break

    acc_valid_records = sorted(acc_valid_records, reverse=True)[:1]
    
    print(acc_valid_records)
    state = {
                "epoch": i + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
    save_path = os.path.join(
        writer.file_writer.get_logdir(),
        "{}_{}_final_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
    )
    if not cfg.get("no_save", False):
        torch.save(state, save_path)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    cfg = AttrDict()

    ##############################################################################
    from exp_helper import init
    init([cfg, ])
    ##############################################################################

    run_id = random.randint(1, 100000)
    logdir = cfg.checkpoint
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))

    from qsparse.util import logging

    logger = logging
    logger.info("Let the games begin")

    train(cfg, writer, logger)
