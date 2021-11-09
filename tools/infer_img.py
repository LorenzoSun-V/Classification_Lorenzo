import numpy as np
import torch
import argparse
from utils.model_utils import *
from models.Mobilenet import mobilenet_v2


label_nh = {'0': 'bank_staff_vest', '1': 'cleaner', '2': 'money_staff', '3': 'person', '4': 'security_staff', '5': 'bank_staff_shirt', '6': 'bank_staff_coat'}
label_intel = {"0": "buildings", "1": "forest", "2": "glacier", "3": "mountain", "4": "sea", "5": "street"}
label_shyh = {'0': 'bank_staff_summer', '1': 'bank_staff_fall', '2': 'custom', '3': 'security_staff'}
label_shsq = {'0': 'person', '1': 'fall_person', '2': 'crouch_person'}


def arg_define():
    parser = argparse.ArgumentParser(description='Classification model train')
    parser.add_argument('--yml', type=str, default='../cfg/mobilenet_v2/nh_bs1024.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    if len(cfg.test.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.test.gpu
    else:
        raise RuntimeError("please use single gpu, change yaml file param: test.gpu")
    img = cv2.imread("/mnt/shy/农行POC/abc_models/staff_cls/demo/samples/0002_94140_1.jpg")
    model_ = eval(f"{cfg.model.model_name}({cfg.model})")
    load_weights(model_, cfg.test.model_path)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()
    task_name = cfg.dataset.task_name
    if task_name == "intel":
        label = label_intel
    elif task_name == "nh":
        label = label_nh
    elif task_name == "shyh":
        label = label_shyh
    elif task_name == "shsq":
        label = label_shsq
    else:
        raise RuntimeError('Unsupported task name right now')
    img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
    # 由于opencv读入的图片都是BGR，在做完图像增强后，把img转成RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_scaler = (img / 255 - 0.5) / 0.5
    img_trans = img_scaler.transpose(2, 0, 1).astype(np.float32)
    img_trans = img_trans[np.newaxis, :, :, :]
    img_tensor = torch.from_numpy(img_trans)
    inputs = img_tensor.to(device=device, non_blocking=True)
    logits = model_(inputs)
    logits = torch.softmax(logits, dim=-1)
    preds = logits.topk(1)[1]
    preds = preds.squeeze(1).item()
    print(label[str(preds)])
    print(logits.cpu().detach().numpy()[0][preds])



