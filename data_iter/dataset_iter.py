from PIL import Image
import torch
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from concurrent.futures import ThreadPoolExecutor
from logzero import logger as log
import numpy as np
import random


class DataIter(Dataset):
    def __init__(self, args, dataset, split="train"):
        super(DataIter, self).__init__()
        self.args = args
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = Image.open(img_path)

        if self.split == "train":
            img_tensor = self.transform_train(img)
        elif self.split == "val":
            img_tensor = self.transform_val(img)
        else:
            raise RuntimeError("data_iter type is train or val")

        target = torch.from_numpy(np.array(label, dtype=int))
        return img_tensor, target

    def transform_train(self, img):
        transform_list = []
        if self.args.is_aug:
            if min(self.args.SCALE[0], self.args.SCALE[1]) > 0 and random.uniform(0, 1) < 0.5:
                transform_list.append(transforms.RandomResizedCrop(size=(self.args.HW[0], self.args.HW[1]),
                                                                   scale=(self.args.SCALE[0], self.args.SCALE[1])))
            if self.args.HFLIP and random.uniform(0, 1) < 0.5:
                transform_list.append(transforms.RandomHorizontalFlip())
            if self.args.COLOR_JITTER > 0 and random.uniform(0, 1) < 0.5:
                transform_list.append(transforms.ColorJitter(brightness=self.args.COLOR_JITTER,
                                                             contrast=self.args.COLOR_JITTER,
                                                             saturation=self.args.COLOR_JITTER))
            if self.args.ROTATE > 0 and random.uniform(0, 1) < 0.5:
                transform_list.append(transforms.RandomRotation(degrees=self.args.ROTATE))
        transform_list.append(transforms.Resize(size=self.args.HW))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        composed_transforms = transforms.Compose(transform_list)
        return composed_transforms(img)

    def transform_val(self, img):
        composed_transforms = transforms.Compose([transforms.Resize(size=self.args.HW),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return composed_transforms(img)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(cfg, dataiter, local_rank):
    assert cfg.train.batch_size % cfg.ddp.NPROCS == 0, "batch size % nprocs is not 0"
    batch_size = int(cfg.train.batch_size / cfg.ddp.NPROCS)
    nw = min([os.cpu_count() // cfg.ddp.WORLD_SIZE, batch_size if batch_size > 1 else 0, 8])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataiter) if local_rank != -1 else None
    dataloader = DataLoaderX(dataiter,
                             batch_size=batch_size,
                             num_workers=nw,
                             sampler=sampler,
                             pin_memory=True)
    return dataloader
