import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset


class LoadImgLabel(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.args = args
        img_dir = Path(args.dataset.train_dir)
        if not args.dataset.train_val_split:
            args.dataset.train_val_split_ratio = 0
        self._check_path([img_dir])
        self.total = self._loadGT(img_dir)
        self.train, self.val = self._train_val_split(self.total, split_ratio=args.dataset.train_val_split_ratio)
        self.label_to_count = self._count_class(self.total)

    def _count_class(self, train_dataset):
        y = np.array(train_dataset, dtype=object)[:, 1:]
        y = np.array([x[0] for x in y])
        if self.args.ONE_HOT:
            label_to_count = Counter(map(tuple, y))
        else:
            label_to_count = Counter(y)

        print("classes statistics:")
        print("  ------------------------------")
        print("  class  | # images")
        print("  ------------------------------")
        for k, v in label_to_count.items():
            print("  {}    | {:5d} ".format(k, v))
        print("  ------------------------------")

        return label_to_count

    def _loadGT(self, img_dir):
        sub_dirs = self._find_classes(img_dir)
        dataset = self._make_dataset(sub_dirs)
        return dataset

    def _find_classes(self, img_dir):
        sub_dirs = [x for x in img_dir.iterdir() if x.is_dir()]
        sub_dirs.sort()
        return sub_dirs

    def _make_dataset(self, sub_dirs):
        dataset = []
        for i, sub_dir in enumerate(tqdm(sub_dirs)):
            img_paths = self._get_imgs(sub_dir)
            for img_path in img_paths:
                label = int(str(sub_dir.name).split('-')[0])
                if self.args.ONE_HOT:
                    one_hot_label = np.zeros(shape=len(sub_dirs))
                    one_hot_label[label] = 1
                    label = one_hot_label
                dataset.append((str(img_path), label))
        return dataset

    def _get_imgs(self, img_dir):
        extentions = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
        img_paths = []
        for ext in extentions:
            try:
                img_paths += glob.glob(str(img_dir / "*.{}".format(ext)))
            except:
                raise RuntimeError("folder not found")
        return img_paths

    def _train_val_split(self, train_dataset, split_ratio):
        X = np.array(train_dataset, dtype=object)[:, 0]
        y = np.array(train_dataset, dtype=object)[:, 1]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_ratio, random_state=42)
        train = np.column_stack((X_train, y_train))
        val = np.column_stack((X_valid, y_valid))

        num_total_imgs = len(self.total)
        num_train_imgs = train.shape[0]
        num_val_imgs = val.shape[0]
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset     | # images")
        print("  ------------------------------")
        print("  total      | {:5d} ".format(num_total_imgs))
        print("  train      | {:5d} ".format(num_train_imgs))
        print("  val        | {:5d} ".format(num_val_imgs))
        print("  ------------------------------")
        return train, val

    def _check_path(self, paths_list):
        for path in paths_list:
            if not Path(path).is_dir():
                raise RuntimeError("'{}' is not a dir".format(path))


if __name__ == "__main__":
    from utils.model_utils import read_yml

    args = read_yml("/mnt/shy/sjh/classify_model/cfg/mobilenet_v2/nh_bs1024.yml")
    data = Dataset(args=args, img_dir="/mnt/shy/sjh/农行data/小图修正9.29/train_all_7")
