# RepVGG: Making VGG-style ConvNets Great Again (MegEngine)

*The PyTorch implementation and pretrained models are available at*
*https://github.com/DingXiaoH/RepVGG*

This is a super simple ConvNet architecture that achieves over 80% top-1 accuracy on ImageNet with a stack of 3x3 conv and ReLU! This repo contains the pretrained models, code for building the model, training, and the conversion from training-time model to inference-time.


| Model | top1 acc | top5 acc | URL(prefix=`https://data.megengine.org.cn/models/weights/`) |
| --- | :---: | :---: | :---: |
| RepVGG-A0    |72.456|90.700|prefix + `repvgg/RepVGG-A0-train.pkl`
| RepVGG-A1    |74.166|91.730|prefix + `repvgg/RepVGG-A1-train.pkl`
| RepVGG-A2    |||prefix + `repvgg/RepVGG-A2-train.pkl`
| RepVGG-B0    |||prefix + `repvgg/RepVGG-B0-train.pkl`
| RepVGG-B1    |||
| RepVGG-B1g2  |||
| RepVGG-B1g4  |||
| RepVGG-B2    |||
| RepVGG-B2g2  |||
| RepVGG-B2g4  |||
| RepVGG-B3    |||
| RepVGG-B3g2  |||
| RepVGG-B3g4  |||

*(More pretrained models comming soon...)*

## Quick start

### 0. prerequisite

Install [MegEngine](https://megengine.org.cn), a deep learning framework developed and used by [Megvii Inc.](https://megvii.com).

```
pip3 install megengine -f https://megengine.org.cn/whl/mge.html
```

### 1. inference on a test image

```
# download pretrained RepVGG-A0 model
wget https://data.megengine.org.cn/models/weights/repvgg/RepVGG-A0-deploy.pkl -O RepVGG-A0-deploy.pkl

# test on an image of cat
python3 inference.py -i assets/cat.jpg --arch RepVGG-A0 --model RepVGG-A0-deploy.pkl
```

### 2. test on ImageNet
```
# download pretrained RepVGG-A0 model
wget https://data.megengine.org.cn/models/weights/repvgg/RepVGG-A0-train.pkl -O RepVGG-A0-train.pkl

# test on ImageNet (also save the converted model)
python3 convert_and_test.py -d [path-to-imagenet] --arch RepVGG-A0 --model RepVGG-A0-train.pkl
```

### 3. train your model on ImageNet
```
# train a RepVGG-A0 on ImageNet
python3 train.py -d [path-to-imagenet] --arch RepVGG-A0
```

<!-- ### 4. benchmark deploy model -->

# Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80\% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.

![image](https://github.com/DingXiaoH/RepVGG/blob/main/arch.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/speed_acc.PNG)
![image](https://github.com/DingXiaoH/RepVGG/blob/main/table.PNG)
