import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
from easydict import EasyDict as edict
from models.load_weights import load_weights
from loss.loss import *
from collections import OrderedDict

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def load_model(model, pretrained_state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
                       k in model_dict and model_dict[k].size() == pretrained_state_dict[k].size()}
    model.load_state_dict(pretrained_dict, strict=False)
    if len(pretrained_dict) == 0:
        print("[INFO] No params were loaded ...")
    else:
        for k, v in pretrained_state_dict.items():
            if k in pretrained_dict:
                print("==>> Load {} {}".format(k, v.size()))
            else:
                print("[INFO] Skip {} {}".format(k, v.size()))
    return model


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
        深度可分离卷积：
        1、convolution on each input channel
        2、Batch Normalize
        3、Relu6
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 深度可分离卷积，卷积的groups等于卷积输入的通道数，实现每个通道单独进行卷积的目的
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)#relu6
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # 有两种卷积stride模式，stride=1和stride=2
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        # 当stride=1且输入、输出通道数相等时，使用short-cut残差连接
        # 当stride=2时不使用short-cut残差连接
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw 使用扩展因子提升通道数
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw， 深度可分离卷积，卷积groups等于输入通道数
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear， 使用1x1卷积降维
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            # stride=1且输入输出维度相等，使用残差连接
            return x + self.conv(x)
        else:
            # 不使用残差连接
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=2,
                 width_mult=1.0,
                 criterion=None,
                 img_size=192,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        assert img_size % 32 == 0
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # first 3x3 convolution
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # 在每个反转卷积中，除第一个卷积外其他卷积操作均使用stride=1
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers, 使用1x1卷积降低通道数
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        # 底层CNN进行特征提取
        self.features = nn.Sequential(*features)

        # building classifier
        # 构建上层分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self.criterion = criterion

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward_train(self, x, target):
        x = self._forward_impl(x)
        train_loss = self.criterion(x, target)
        return x, train_loss

    def forward_test(self, x):
        return self._forward_impl(x)

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)


def mobilenet_v2(cfg):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    cfg = edict(cfg)
    criterion = choose_loss(cfg.loss, cfg.num_classes)
    if cfg.bn == 'bn':
        norm_layer = None
    elif cfg.bn == 'syncbn':
        norm_layer = nn.SyncBatchNorm
    else:
        raise RuntimeError('Unknown bn type, check param model.bn in yaml. Only support bn and syncbn')
    model = MobileNetV2(num_classes=cfg.num_classes, width_mult=1.0, criterion=criterion, norm_layer=norm_layer)
    if cfg.pretrained:
        # 加载在ImageNet上预训练的参数
        load_weights(model, cfg.pretrained_model_url)
    return model


if __name__ == "__main__":
    from utils.model_utils import read_yml
    cfg = read_yml('/mnt/shy/sjh/classify_model/cfg/mobilenet_v2/nh_bs1024.yml')
    input = torch.randn([4, 3, 192, 192])
    gt_ = torch.from_numpy(np.array([1, 2, 4, 5]))
    model = mobilenet_v2(cfg.model)
    output = model(input)
    print(output)
