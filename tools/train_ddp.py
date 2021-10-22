import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import random
import numpy as np
import logzero
from logzero import logger as log

from utils.model_utils import *
from data_iter.load_img_label import LoadImgLabel
from data_iter.dataset_iter import DataIter, create_dataloader
from engine import train_one_epoch, val_one_epoch
from models.Mobilenet import mobilenet_v2


def arg_define():
    parser = argparse.ArgumentParser(description='Classification model train')
    parser.add_argument('--yml', type=str, default='../cfg/mobilenet_v2/nh_bs1024.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if args.USE_DDP:
            self._init_process()
        self.load_device()
        self.load_model()
        self.load_optim()

    def _init_process(self):
        dist.init_process_group(backend=self.args.ddp.DIST_BACKEND,
                                init_method=self.args.ddp.DIST_URL,
                                rank=self.args.ddp.LOCAL_RANK,
                                world_size=self.args.ddp.WORLD_SIZE)

    def load_device(self):
        if self.args.USE_DDP:
            if self.args.ddp.LOCAL_RANK in [0, -1]:
                log.info("=> load gpu device")
            torch.cuda.set_device(self.args.ddp.LOCAL_RANK)
            self.device = self.args.ddp.LOCAL_RANK
        return self

    def load_model(self):
        if self.args.ddp.LOCAL_RANK in [0, -1]:
            log.info("=> load model")
        model = eval(f"{self.args.model.model_name}({self.args.model})")
        if self.args.USE_DDP:
            self.model = DDP(model.cuda(self.device), device_ids=[self.device], output_device=self.device, find_unused_parameters=True)
        return self

    def load_optim(self):
        if self.args.train.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif self.args.train.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             self.args.train.base_lr,
                                             momentum=0.9,
                                             weight_decay=1e-4)
        else:
            log.error(f"wrong config {self.args.train.optim}")
        if self.args.train.warmup_cosdecay > 0:
            lr_lambda = cal_lr_lambda(self.args.train.total_epochs, self.args.train.warmup_cosdecay)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def run(self, epochs, train_sampler, train_dataloader, val_dataloader):
        best_acc = 0
        best_loss = 100

        for epoch in range(1, epochs+1):
            train_sampler.set_epoch(epoch)
            if self.args.ddp.LOCAL_RANK in [0, -1]:
                log.info(f"TRAIN | Epoch: [{epoch}/{epochs}] | LR: {self.optimizer.param_groups[0]['lr']:.3f} \n")
            train_loss, train_prec = train_one_epoch(train_dataloader, self.model, self.optimizer, self.device,
                                                     self.args.train.print_freq, epoch, epochs, self.args.ONE_HOT,
                                                     self.args.ddp.LOCAL_RANK)
            self.scheduler.step()
            if val_dataloader is not None:
                print(f"this is process {self.args.ddp.LOCAL_RANK} do validation")
                val_loss, val_prec = val_one_epoch(val_dataloader, self.model, self.device, self.args.ONE_HOT, self.args.ddp.LOCAL_RANK)

                acc_is_best = val_prec >= best_acc
                best_acc = max(val_prec, best_acc)
                if acc_is_best:
                    torch.save(self.model.state_dict(), os.path.join(self.args.LOG_DIR, "model_best_acc.pth"))
                    log.info(f"save model_best_acc_{best_acc}")
                    log.info(" ********************************************* \n")

                loss_is_best = val_loss <= best_loss
                best_loss = min(val_loss, best_loss)
                if loss_is_best:
                    torch.save(self.model.state_dict(), os.path.join(self.args.LOG_DIR, "model_best_loss.pth"))
                    log.info(f"save model_best_loss_{best_loss}")
                    log.info(" ********************************************* \n")

                for i in self.args.train.checkpoint_epochs:
                    if epoch == i:
                        torch.save(self.model.state_dict(), os.path.join(self.args.LOG_DIR, f"model_{epoch}.pth"))

        if self.args.ddp.LOCAL_RANK in [0, -1]:
            log.info("End trainning,Save model_final \n")
            model_file = os.path.join(self.args.LOG_DIR, f"model_final_{epoch}.pth")
            torch.save(self.model.state_dict(), model_file)
            return self


def main_worker(local_rank, nprocs, cfg, train_dataiter, val_dataiter, log_path):
    cfg.ddp.WORLD_SIZE = nprocs * cfg.ddp.WORLD_SIZE
    cfg.ddp.LOCAL_RANK = local_rank
    set_seed(local_rank, cfg.SEED)
    if local_rank in [-1, 0]:
        logzero.logfile(log_path, maxBytes=1e6, backupCount=3)

    log.info(f"this is {local_rank}")
    trainer = Trainer(cfg)
    train_sampler, train_dataloader = create_dataloader(cfg, train_dataiter, cfg.ddp.LOCAL_RANK)
    if local_rank in [-1, 0]:
        _, val_dataloader = create_dataloader(cfg, val_dataiter, -1)
    else:
        val_dataloader = None
    trainer.run(cfg.train.total_epochs, train_sampler, train_dataloader, val_dataloader)
    print(local_rank)


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    cfg, log_path = create_log_dir(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_IDS
    torch.backends.cudnn.benchmark = True

    log.info("=> load data")
    dataset = LoadImgLabel(cfg, cfg.dataset.train_dir)
    train_dataiter = DataIter(cfg, dataset.train, is_train=True)
    val_dataiter = DataIter(cfg, dataset.val, is_train=False)
    if cfg.USE_DDP:
        cfg.ddp.NPROCS = len(cfg.DEVICE_IDS.split(","))
        mp.spawn(main_worker, nprocs=cfg.ddp.NPROCS, args=(cfg.ddp.NPROCS, cfg, train_dataiter, val_dataiter, log_path))
    else:
        print("train_ddp is train with ddp mode, use train and turn off ddp mode if only have single GPU ")

