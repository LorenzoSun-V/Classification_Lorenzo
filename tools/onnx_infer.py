import os
import cv2
import numpy as np
import glob


def infer_img(img_dir, net):
    if type(img_dir) == str:
        #  如果传入的是图片路径，那么就按照图片方式处理
        srcimg = cv2.imread(img_dir)
    else:
        #  如果传入的是图片np数组，那么就按照video方式处理
        srcimg = img_dir
    # 前处理
    img = cv2.resize(srcimg, (128, 256), interpolation=cv2.INTER_CUBIC)
    img = (img / 255 - 0.5) / 0.5
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 模型推理
    blob = cv2.dnn.blobFromImage(img)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    print(outs)

    return outs


def load_onnx(is_video):
    onnx_path = "/mnt/shy/sjh/classify_model/model_exp/mobilenet_v2/shyh/shyh_cls_v0.onnx"
    try:
        net = cv2.dnn.readNet(onnx_path)
        print('read sucess')
    except:
        print('read failed')
    if not is_video:
        img_root_path = "/mnt/shy/sjh/上海银行data/test/0-bank_staff_summer"
        img_dirs = glob.glob(os.path.join(img_root_path, "*.jpg"))
        for img_dir in img_dirs:
            img_result = infer_img(img_dir, net)
            print(img_result)


if __name__ == "__main__":
    load_onnx(is_video=False)
