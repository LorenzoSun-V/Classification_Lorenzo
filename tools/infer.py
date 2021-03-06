import torch
from torch.utils.data import DataLoader
import shutil
import argparse
from tqdm import tqdm
from utils.model_utils import *
from models.Mobilenet import mobilenet_v2
from data_iter.load_img_label import LoadImgLabel
from data_iter.dataset_iter import DataIter, create_dataloader


label_nh = {'0': 'bank_staff_vest', '1': 'cleaner', '2': 'money_staff', '3': 'person', '4': 'security_staff', '5': 'bank_staff_shirt', '6': 'bank_staff_coat', '7': 'security_staff_black'}
label_intel = {"0": "buildings", "1": "forest", "2": "glacier", "3": "mountain", "4": "sea", "5": "street"}
label_shyh = {'0': 'bank_staff_summer', '1': 'bank_staff_fall', '2': 'custom', '3': 'security_staff'}
label_shsq = {'0': 'person', '1': 'fall_person', '2': 'crouch_person'}


def arg_define():
    parser = argparse.ArgumentParser(description='Classification model train')
    parser.add_argument('--yml', type=str, default='../cfg/mobilenet_v2/intel_bs1024.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


def eval_softmax(model, loader, test_imgs_num, num_classes, label):
    model.eval()
    running_corrects = 0
    confusion_matrix = np.zeros((num_classes, num_classes)).astype(np.uint16)

    for batch_id, (inputs, targets) in enumerate(tqdm(loader)):
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        logits = model(inputs)
        logits = torch.softmax(logits, dim=-1)
        preds = logits.topk(1)[1]
        # all_results.append([preds.cpu().item(), targets.cpu().item()])
        preds = preds.squeeze(1)
        for (pred, target) in zip(preds, targets):
            confusion_matrix[target, pred] += 1
        positive = (preds == targets)
        running_corrects += torch.sum(positive).item()

    # results = np.zeros(len(LABELS))
    # for pred, target in all_results:

    label_class = [label[i] for i in label]
    plot_confusion_matrix(confusion_matrix, classes=label_class, normalize=False, title='confusion matrix')
    overall_acc = running_corrects / test_imgs_num
    print("Overall Acc: {:.4f}".format(overall_acc))


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    if len(cfg.test.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.test.gpu
    else:
        raise RuntimeError("please use single gpu, change yaml file param: test.gpu")

    dataset = LoadImgLabel(cfg, cfg.test.test_dir)
    test_dataiter = DataIter(cfg.aug, dataset.total, split="val")
    test_dataloader = DataLoader(test_dataiter,
                                 batch_size=cfg.test.batch_size,
                                 pin_memory=True,
                                 shuffle=False)
    weight = None
    model_ = eval(f"{cfg.model.model_name}({cfg.model}, weight)")
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
    eval_softmax(model_, test_dataloader, len(dataset.total), cfg.model.num_classes, label)


