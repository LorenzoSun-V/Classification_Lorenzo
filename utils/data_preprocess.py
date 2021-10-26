import os
import glob
from pathlib import Path


label_dict = {"buildings": "0", "forest": "1", "glacier": "2", "mountain": "3", "sea": "4", "street": "5"}


def folder_rename(root_path):
    #  rename folders with label
    for dir_path in glob.glob(os.path.join(root_path, '*')):
        dir_path = Path(dir_path)
        dir_name = dir_path.name
        label = label_dict[dir_name]
        new_dir_name = f"{label}-{dir_name}"
        new_dir_path = dir_path.parent / new_dir_name
        print(new_dir_path)
        dir_path.rename(new_dir_path)


if __name__ == "__main__":
    folder_rename("/mnt/shy/sjh/classify_model/datasets/intel/seg_test")
