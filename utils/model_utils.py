import os
import math
import cv2
import logging
import json
from pprint import pprint
import logzero
import yaml
import time
import numpy as np
from logzero import logger as log
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
from easydict import EasyDict as edict
from pathlib import Path, PosixPath
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools
# from utils.loss import *
import warnings


def set_seed(local_rank, seed=666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    if local_rank in [-1, 0]:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def checkfolder(paths):
    if isinstance(paths, str):
        if not Path(paths).is_dir():
            os.mkdir(paths)
            log.info("Created new directory in %s" % paths)

    if isinstance(paths, PosixPath):
        if not Path(paths).is_dir():
            paths.mkdir(parents=True)
            # Path.mkdir(paths)
            log.info("Created new directory in %s" % paths)


def read_yml(yml_file):
    with open(yml_file) as f:
        cfg = edict(yaml.safe_load(f))
    return cfg


def train_val_split(train_img_path, train_mask_path, val_factor):
    log.info('=> train val split ')

    train_img_path = Path(train_img_path)
    train_mask_path = Path(train_mask_path)
    img_paths = [path for path in train_img_path.iterdir() if path.suffix == '.jpg' or path.suffix == '.png']
    length = len(img_paths)
    val_size = int(length*val_factor)
    random.shuffle(img_paths)
    val_img = img_paths[:val_size]
    train_img = img_paths[val_size:]
    val_mask = [train_mask_path / "".join((path.stem, ".png")) for path in val_img]
    train_mask = [train_mask_path / "".join((path.stem, ".png")) for path in train_img]
    return train_img, train_mask, val_img, val_mask


def cal_lr_lambda(epochs, warmup_cos_decay):
    # learning rate warmup and cosine decay
    t = warmup_cos_decay
    lambda1 = lambda epoch: ((epoch+1) / t) if (epoch+1) <= t else 0.1 \
        if 0.5 * (1+math.cos(math.pi*(epoch+1-t) / (epochs-t))) < 0.1 else 0.5 * (1+math.cos(math.pi*(epoch+1-t)/(epochs-t)))
    return lambda1


def create_log_dir(cfg):
    # print train set args, create pth and log dir
    local_time = time.localtime()
    if cfg.LOG_DIR == "":
        # create folder saving log files and pth files adaptively via model_exp & model_name & task_name
        # path like:  /model_exp/model_name/task_name/2021-09-28_13-47-26
        pth_path = Path(cfg.dataset.model_exp)
        pth_path = pth_path / cfg.model.model_name / cfg.dataset.task_name / time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
        cfg.LOG_DIR = str(pth_path)
    else:
        # create folder via LOG_DIR
        pth_path = Path(cfg.LOG_DIR)
    checkfolder(pth_path)

    log_path = str(pth_path / "log.txt")
    logzero.logfile(log_path, maxBytes=1e6, backupCount=3)
    log.info("################################################ NEW LOG")
    pprint(cfg)
    fs = open(pth_path / 'train_ops.json', "w", encoding='utf-8')
    json.dump(cfg, fs, ensure_ascii=False, indent=1)
    fs.close()

    return cfg, log_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_fscore(y_pred, y_true):
    eps = torch.FloatTensor([1e-7])
    beta = torch.FloatTensor([1])

    true_positive = (y_pred * y_true).sum(dim=0)
    precision = true_positive.div(y_pred.sum(dim=0).add(eps)) ##p = tp / (tp + fp + eps)
    recall = true_positive.div(y_true.sum(dim=0).add(eps)) ##r = tp / (tp + fn + eps)
    micro_f1 = torch.mean((precision*recall).div(precision.mul(beta).mul(beta) + recall + eps).mul(1 + beta).mul(1+beta))

    return precision, recall, micro_f1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def load_weights(model: nn.Module, model_url: str):
    state_dict = torch.load(model_url, map_location=lambda storage, loc:storage)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(f'Error: The pretrained weights from "{model_url}" cannot be loaded')
        exit(0)
    else:
        print(f'Successfully loaded imagenet pretrained weights from {model_url}')
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                f'due to unmatched keys or layer size: {discarded_layers}')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    # train_img, train_mask, val_img, val_mask = train_val_split("/mnt/shy/sjh/YOLOP-main/hedao_image_30", "/mnt/shy/sjh/YOLOP-main/hedao_image_30_mask", 0.1)
    lambda1 = cal_lr_lambda(120, 10)
    print(lambda1(10))



