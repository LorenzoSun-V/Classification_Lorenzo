import os
import glob
from pathlib import Path


label_intel = {"buildings": "0", "forest": "1", "glacier": "2", "mountain": "3", "sea": "4", "street": "5"}
label_nh = {'bank_staff_vest': '0', 'cleaner': '1', 'money_staff': '2', 'person': '3', 'security_staff': '4', 'bank_staff_shirt': '5', 'bank_staff_coat': '6'}
label_shyh = {'bank_staff_summer': '0', 'bank_staff_fall': '1', 'custom': '2', 'security_staff': '3'}
label_shsq = {'person': '0', 'fall_person': '1', 'crouch_person': '2'}


def folder_rename(root_path, label_dict):
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
    label_dict = label_nh
    folder_rename("/mnt/shy/sjh/农行data/测试数据/第四批", label_dict)
