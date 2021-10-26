# Classification Model with Pytorch

## Quick start
**Note : Use Python 3.6 or newer**

　　　**Use Pytorch 1.7.1 or newer**

1. [Download kaggle intel datasets](https://www.kaggle.com/puneet6060/intel-image-classification)
2. Use [folder_rename](https://github.com/Gakkkkkki/Classification_Lorenzo/tree/master/utils/data_preprocess.py) in utils/data_preprocess.py to label each class folder.

　　The train/test dataset directory structure is like following(folder name: label-class):
```
.
├── seg_train
│   ├── 0-buildings
│   │   ├── 0.jpg
│   │   ├── 4.jpg
│   │   ├── ...
│   │   └── 20054.jpg
│   ├── 1-forest
│   │   ├── 8.jpg
│   │   ├── 23.jpg
│   │   ├── ...
│   │   └── 20051.jpg
│   ...
│   ├── 5-street
│   │   ├── 8.jpg
│   │   ├── 23.jpg
│   │   ├── ...
│   │   └── 20051.jpg
└── seg_test
    ├── 0-buildings
    │   ├── 20057.jpg
    │   ├── 20060.jpg
    │   ├── ...
    │   └── 24322.jpg
    ├── 1-forest
    │   ├── 20056.jpg
    │   ├── 20062.jpg
    │   ├── ...
    │   └── 24324.jpg
    ...
    └── 5-street
        ├── 20066.jpg
        ├── 20067.jpg
        ├── ...
        └── 24332.jpg

```

### Use MobileNetV2 as an example, more models will be updated.

3. Train (Use absolute path if have any path error)

- Using single GPU:
    
　　modify yaml file path in [tools/train.py](https://github.com/Gakkkkkki/Classification_Lorenzo/blob/master/tools/train.py) as './cfg/mobilenet_v2/intel_bs256.yml' and run train.py in IDE.

　　or run in terminal directly:
```terminal
　python tools/train.py --yml ./cfg/mobilenet_v2/intel_bs256.yml
```
- Using multiple GPUs(DDP train mode):

　　modify yaml file path in [tools/train.py](https://github.com/Gakkkkkki/Classification_Lorenzo/blob/master/tools/train.py) as './cfg/mobilenet_v2/intel_bs1024.yml' and run train.py in IDE.

4. Infer (Use absolute path if have any path error)

　　modify test.model_path as your weights path in yaml file you use.　
    
　　modify yaml file path in [tools/infer.py](https://github.com/Gakkkkkki/Classification_Lorenzo/blob/master/tools/infer.py) as './cfg/mobilenet_v2/intel_bs256.yml' and run infer.py

　　or run in terminal directly:
```terminal
　python tools/infer.py --yml ./cfg/mobilenet_v2/intel_bs256.yml
```
