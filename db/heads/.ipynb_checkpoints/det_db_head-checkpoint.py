# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def get_bias_attr(fan_out):
    # 对应 Paddle 的 get_bias_attr 函数
    # stdv = 1.0 / math.sqrt(k * 1.0) 其中 k 是 fan_out
    stdv = 1.0 / math.sqrt(fan_out)
    # 返回一个初始化函数，用于在层创建后调用
    def init_bias(m):
        m.bias.data.uniform_(-stdv, stdv)
    return init_bias


class Head(nn.Module):
    def __init__(self, in_channels, name_list):
        super(Head, self).__init__()
        
        # 1. Conv2D -> Conv2d
        #    bias_attr=False 对应 bias=False
        #    weight_attr=ParamAttr() 对应默认初始化，PyTorch Conv2d 默认就是Kaiming Uniform
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False) # PyTorch 默认有 bias，需显式关闭
        
        # 2. BatchNorm
        #    num_channels -> num_features
        #    param_attr (weight) 初始化为 Constant(1.0)
        #    bias_attr (bias) 初始化为 Constant(1e-4)
        #    act='relu' -> nn.ReLU(inplace=True)
        self.conv_bn1 = nn.Sequential(
            nn.BatchNorm2d(
                num_features=in_channels // 4,
                # PyTorch BatchNorm 的 weight 和 bias 默认就是 1 和 0
                # 我们在 reset_parameters 或 __init__ 后手动修改
            ),
            nn.ReLU(inplace=True)
        )
        
        # 3. Conv2DTranspose -> ConvTranspose2d
        #    weight_attr 使用 KaimingUniform -> 使用 kaiming_uniform_ 初始化
        #    bias_attr 使用 get_bias_attr -> 需在 reset 中调用
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2,
            bias=True # 默认为 True，这里显式写出
        )
        
        self.conv_bn2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 4. 最后的 Transpose
        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2,
            bias=True
        )

        # 5. 手动进行权重初始化 (替代 Paddle 的 weight_attr / bias_attr)
        self._initialize_weights(in_channels)

    def _initialize_weights(self, in_channels):
        # 初始化 conv1 (BN 通常不需要手动初始化 weight/bias，但为了严格对齐)
        init.constant_(self.conv_bn1[0].weight, 1.0)
        init.constant_(self.conv_bn1[0].bias, 1e-4)
        
        # 初始化 conv2 (Transposed)
        # PyTorch 默认初始化通常是 Kaiming Uniform，这里显式调用以确保
        # bias 初始化
        get_bias_attr(in_channels // 4)(self.conv2)
        
        # 初始化 conv_bn2
        init.constant_(self.conv_bn2[0].weight, 1.0)
        init.constant_(self.conv_bn2[0].bias, 1e-4)
        
        # 初始化 conv3 (Transposed)
        get_bias_attr(in_channels // 4)(self.conv3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = torch.sigmoid(x) # F.sigmoid 已弃用，推荐直接使用 torch.sigmoid
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        # 注意：PyTorch 不需要 name_list 来定义参数名，参数名由 Module 的属性名自动确定
        # 这里保留变量但实际代码中不使用
        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48',
            'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'conv2d_transpose_2', 'batch_norm_50',
            'conv2d_transpose_3', 'thresh'
        ]
        
        # 直接实例化 Head
        self.binarize = Head(in_channels, binarize_name_list)
        self.thresh = Head(in_channels, thresh_name_list)

    def step_function(self, x, y):
        # 对应 paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))
        # 即 1 / (1 + exp(-k*(x-y)))
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        
        # PyTorch 中通常用 'training' 属性判断，或者显式传参
        # 这里逻辑保持一致
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        
        # 拼接 dim=1 (Channel 维度)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {'maps': y}