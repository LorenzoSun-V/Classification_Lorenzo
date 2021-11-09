import os
import argparse
import onnxruntime
import torch
from models.Mobilenet_softmax import mobilenet_v2
from models.load_weights import load_weights
from utils.model_utils import read_yml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def arg_define():
    parser = argparse.ArgumentParser(description='Classification model train')
    parser.add_argument('--yml', type=str, default='../cfg/mobilenet_v2/shsq_bs512_256x128.yml', help='path of cfg file')
    args = parser.parse_args()
    return args


def to_onnx(model, onnx_path, H, W):
    data = torch.randn(16, 3, H, W)
    data = data.to(device=torch.device("cuda"), non_blocking=True)
    torch.onnx.export(model, data, onnx_path, verbose=True)
    print( f"**Completed creating onnx path {onnx_path}" )


if __name__ == "__main__":
    args = arg_define()
    cfg = read_yml(args.yml)
    if len(cfg.test.gpu) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.test.gpu
    model_ = eval(f"{cfg.model.model_name}({cfg.model})")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if cfg.USE_DDP:
        load_weights(model_, cfg.test.model_path)
    else:
        ckpt = torch.load(cfg.test.model_path, map_location=device)
        model_.load_state_dict(ckpt)

    model_ = model_.to(device)
    model_.eval()

    onnx_name = cfg.test.model_path.split("/")[-1].replace("pth", "onnx")
    onnx_path = os.path.join("/mnt/shy/sjh/classify_model/onnx/mobilenet_v2/shyh", onnx_name)
    # compute ONNX Runtime output prediction
    to_onnx(model_, onnx_path, cfg.aug.HW[0], cfg.aug.HW[1])
