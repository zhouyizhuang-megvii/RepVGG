# MIT License

# Copyright (c) 2020 megvii-model

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import collections

import megengine
import megengine.functional as F
import megengine.module as M
import numpy as np


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    return M.ConvBn2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
    )


class RepVGGBlock(M.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = M.ReLU()

        if deploy:
            self.rbr_reparam = M.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )  # , padding_mode=padding_mode)

        else:
            self.rbr_identity = (
                M.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )
            print("RepVGG Block, identity = ", self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, M.ConvBn2d):
            kernel = branch.conv.weight.numpy()
            running_mean = branch.bn.running_mean.numpy()
            running_var = branch.bn.running_var.numpy()
            gamma = branch.bn.weight.numpy()
            beta = branch.bn.bias.numpy()
            eps = branch.bn.eps
        else:
            assert isinstance(branch, M.BatchNorm2d)
            kernel = np.zeros((self.in_channels, self.in_channels, 3, 3))
            for i in range(self.in_channels):
                kernel[i, i, 1, 1] = 1
            running_mean = branch.running_mean.numpy()
            running_var = branch.running_var.numpy()
            gamma = branch.weight.numpy()
            beta = branch.bias.numpy()
            eps = branch.eps
        # NOTE: megengine beta, gamma is of shape (C,), reshape to torch shape
        beta = beta.reshape(running_mean.shape)
        gamma = gamma.reshape(running_mean.shape)
        std = np.sqrt(running_var + eps)
        t = gamma / std
        t = np.reshape(t, (-1, 1, 1, 1))
        t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        kernel = np.zeros((kernel1x1.shape[0], kernel1x1.shape[1], 3, 3))
        kernel[:, :, 1:2, 1:2] = kernel1x1
        return kernel

    def repvgg_convert(self):
        kernel3x3, bias3x3 = self._fuse_bn(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )


class RepVGG(M.Module):
    def __init__(
        self,
        num_blocks,
        num_classes=1000,
        width_multiplier=None,
        override_groups_map=None,
        deploy=False,
        more_layers_on_112=0,
    ):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        # self.stage0 = REVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)

        if more_layers_on_112 == 0:
            self.stage0 = RepVGGBlock(
                in_channels=3,
                out_channels=self.in_planes,
                kernel_size=3,
                stride=2,
                padding=1,
                deploy=self.deploy,
            )
        elif more_layers_on_112 == 1:
            self.stage0 = M.Sequential(
                RepVGGBlock(
                    in_channels=3,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    deploy=self.deploy,
                ),
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    deploy=self.deploy,
                ),
            )
        elif more_layers_on_112 == 2:
            self.stage0 = M.Sequential(
                RepVGGBlock(
                    in_channels=3,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    deploy=self.deploy,
                ),
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    deploy=self.deploy,
                ),
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    deploy=self.deploy,
                ),
            )

        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = M.AvgPool2d(7)
        self.linear = M.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return M.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = F.flatten(out, 1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_A1(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_A2(deploy=False):
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        num_classes=1000,
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B0(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B1(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B1g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B1g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_B2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B2g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B2g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        deploy=deploy,
    )


def create_RepVGG_B3(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        deploy=deploy,
    )


def create_RepVGG_B3g2(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
        deploy=deploy,
    )


def create_RepVGG_B3g4(deploy=False):
    return RepVGG(
        num_blocks=[4, 6, 16, 1],
        num_classes=1000,
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        deploy=deploy,
    )


func_dict = {
    "RepVGG-A0": create_RepVGG_A0,
    "RepVGG-A1": create_RepVGG_A1,
    "RepVGG-A2": create_RepVGG_A2,
    "RepVGG-B0": create_RepVGG_B0,
    "RepVGG-B1": create_RepVGG_B1,
    "RepVGG-B1g2": create_RepVGG_B1g2,
    "RepVGG-B1g4": create_RepVGG_B1g4,
    "RepVGG-B2": create_RepVGG_B2,
    "RepVGG-B2g2": create_RepVGG_B2g2,
    "RepVGG-B2g4": create_RepVGG_B2g4,
    "RepVGG-B3": create_RepVGG_B3,
    "RepVGG-B3g2": create_RepVGG_B3g2,
    "RepVGG-B3g4": create_RepVGG_B3g4,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


#   Use like this:
#   train_model = create_RepVGG_A0(deploy=False)
#   train train_model
#   deploy_model = repvgg_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
def repvgg_convert(model, build_func, save_path=None):
    converted_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "repvgg_convert"):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + ".rbr_reparam.weight"] = kernel
            converted_weights[name + ".rbr_reparam.bias"] = bias
        elif isinstance(module, M.Linear):
            converted_weights[name + ".weight"] = module.weight.numpy()
            converted_weights[name + ".bias"] = module.bias.numpy()
        else:
            print(name, type(module))
    del model

    deploy_model = build_func(deploy=True)
    for name, param in deploy_model.named_parameters():
        print("deploy param: ", name, param.shape, np.mean(converted_weights[name]))
        param[:] = megengine.tensor(converted_weights[name], dtype="float32")

    if save_path is not None and save_path.endswith("pkl"):
        megengine.save(deploy_model.state_dict(), save_path)

    return deploy_model


if __name__ == "__main__":
    model = create_RepVGG_A0()
    # model.load_state_dict(megengine.load("output/RepVGG-A0/checkpoint.pkl")["state_dict"])
    image = megengine.tensor(np.zeros([2, 3, 224, 224]), dtype="float32") + 128

    model.eval()
    logits = model(image)

    def get_parameters(model):
        params_wd = []
        params_nwd = []
        for n, p in model.named_parameters():
            if n.find("bias") >= 0 or n.find("bn.") >= 0:
                print("NOT include", n, p.shape)
                params_nwd.append(p)
            else:
                print("    include", n, p.shape)
                params_wd.append(p)
        return [
            {"params": params_wd},
            {"params": params_nwd, "weight_decay": 0},
        ]

    get_parameters(model)

    deploy_model = repvgg_convert(model, create_RepVGG_A0)
    deploy_model.eval()
    deploy_logits = deploy_model(image)
    print(logits, deploy_logits)
