import os
from tqdm import tqdm
import torch
import time
import argparse
from data_iter.load_img_label import LoadImgLabel
from data_iter.dataset_iter import DataIter, create_dataloader
from utils.model_utils import read_yml


def arg_define():
    parser = argparse.ArgumentParser(description='Seg model train')
    parser.add_argument('--yml', type=str, default='../cfg/mobilenet_v2/nh_bs1024.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


def calculate_mean_std(cfg, total_dataiter, n_channels):
    _, total_dataloader = create_dataloader(cfg, total_dataiter, -1)
    start = time.time()
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    print('=> Computing mean and std ..')
    for images, masks in tqdm(total_dataloader):
        for i in range(n_channels):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean.div_(total_dataiter.__len__())
    std.div_(total_dataiter.__len__())
    print(mean, std)

    print(f"time elapsed: {time.time() - start}")


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    dataset = LoadImgLabel(cfg)
    total_dataiter = DataIter(cfg, dataset.total, is_train=False, is_cal_std=True)
    calculate_mean_std(cfg, total_dataiter, n_channels=3)

