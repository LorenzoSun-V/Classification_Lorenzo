import cv2
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from concurrent.futures import ThreadPoolExecutor
from logzero import logger as log
import numpy as np
from .aug import *


class DataIter(Dataset):
    def __init__(self, args, dataset, is_train=True, is_cal_std=False):
        super(DataIter, self).__init__()
        self.args = args
        self.dataset = dataset
        self.img_paths = [i[0] for i in dataset]
        self.transform = []
        self.is_train = is_train
        self.is_cal_std = is_cal_std
        if is_train:
            if args.aug.is_aug:
                if args.aug.COLOR_JITTER:
                    self.transform.append(ColorJitter())
                if args.aug.ROTATE > 0:
                    self.transform.append(Rotation(args.aug.ROTATE))
                if args.aug.HFLIP:
                    self.transform.append(Flip(mode='h'))
                if args.aug.CROP:
                    self.transform.append(ScaleCrop(args.aug.HW[0], args.aug.HW[1], args.aug.SCALE))
                else:
                    self.transform.append(Resize(size_in_pixel=(args.aug.HW[1], args.aug.HW[0])))
            else:
                self.transform.append(Resize(size_in_pixel=(args.aug.HW[1], args.aug.HW[0])))
        else:
            self.transform.append(Resize(size_in_pixel=(args.aug.HW[1], args.aug.HW[0])))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print("getitem?")
        img_path, label = self.dataset[index]
        img = cv2.imread(img_path)

        if img.dtype != np.uint8:
            print(f'Input image {img_path} is not uint8')
            return None

        for trans in self.transform:
            # print(trans)
            img = trans(img)
        # 由于opencv读入的图片都是BGR，在做完图像增强后，把img转成RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if not self.is_train:
        #     save_img_path = f"/mnt/shy/sjh/test_aug/aug/{index}_{self.args.ddp.LOCAL_RANK}.jpg"
        #     cv2.imwrite(save_img_path, img)
        if not self.is_cal_std:
            img_scaler = (img / 255 - 0.5) / 0.5
        else:
            img_scaler = img
        img_trans = img_scaler.transpose(2, 0, 1).astype(np.float32)
        img_tensor = torch.from_numpy(img_trans)

        target = torch.from_numpy(np.array(label, dtype=int))
        return img_tensor, target


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(cfg, dataiter, local_rank):
    assert cfg.train.batch_size % cfg.ddp.NPROCS == 0, "batch size % nprocs is not 0"
    batch_size = int(cfg.train.batch_size / cfg.ddp.NPROCS)
    nw = min([os.cpu_count() // cfg.ddp.WORLD_SIZE, batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataiter, shuffle=True) if local_rank != -1 else None
    dataloader = DataLoaderX(dataiter,
                             batch_size=batch_size,
                             num_workers=nw,
                             sampler=train_sampler,
                             pin_memory=True)
    return train_sampler, dataloader
