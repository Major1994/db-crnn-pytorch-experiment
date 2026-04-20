import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果你需要完全一致的 PaddlePaddle 行为，可以使用下面的自定义函数
# PyTorch 1.7+ 有内置的 F.hardswish，但为了兼容性和 Paddle 的精度，这里提供自定义版本
def hard_sigmoid(x, slope=0.2, offset=0.5):
    """
    PaddlePaddle 的 hard_sigmoid 实现。
    默认 slope=0.2, offset=0.5 对应经典的 [0,1] 范围的 HSwish。
    """
    return F.relu6(x * slope + offset) / 6.0

def hard_swish(x):
    """Hard Swish 激活函数"""
    return x * hard_sigmoid(x)

class SEModule(nn.Module):
    """Squeeze-and-Excite 模块"""
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 注意：PyTorch 是 AdaptiveAvgPool2d
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        # 这里使用上面定义的 hard_sigmoid，保持与 Paddle 一致
        outputs = hard_sigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs  


class ConvBNLayer(nn.Module):
    """卷积 + 批归一化 Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, if_act=True, act=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        
        # PyTorch 的 Conv2d 参数顺序与 Paddle 一致
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False) # PyTorch 默认有 bias，这里设为 False 以匹配 BatchNorm

        # PyTorch 的 BatchNorm2d
        self.bn = nn.BatchNorm2d(num_features=out_channels) # 注意：PyTorch 参数是 num_features，不是 num_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                # 使用自定义的 hard_swish
                x = hard_swish(x)
            else:
                print("The activation function({}) is selected incorrectly.".format(self.act))
                exit()
        return x


class ResidualUnit(nn.Module):
    """倒残差单元 (Inverted Residual Block)"""
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, act=None):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2), # PyTorch padding 计算逻辑一致
            groups=mid_channels,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False, # 注意：最后的 Linear Conv 通常不加激活函数
            act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = x + inputs # PyTorch 的 add 操作直接用 + 号，不需要 paddle.add
        return x


class MobileNetV3(nn.Module):
    """MobileNetV3 主干网络"""
    def __init__(self, in_channels=3, model_name='large', scale=0.5, disable_se=False, **kwargs):
        super(MobileNetV3, self).__init__()
        self.disable_se = disable_se

        # 配置表 (Config)
        if model_name == "large":
            cfg = [
                # k, exp,  c,   se,     nl,        s
                [3, 16,   16,  False,  'relu',    1],
                [3, 64,   24,  False,  'relu',    2],
                [3, 72,   24,  False,  'relu',    1],
                [5, 72,   40,  True,   'relu',    2],
                [5, 120,  40,  True,   'relu',    1],
                [5, 120,  40,  True,   'relu',    1],
                [3, 240,  80,  False,  'hardswish', 2],
                [3, 200,  80,  False,  'hardswish', 1],
                [3, 184,  80,  False,  'hardswish', 1],
                [3, 184,  80,  False,  'hardswish', 1],
                [3, 480,  112, True,   'hardswish', 1],
                [3, 672,  112, True,   'hardswish', 1],
                [5, 672,  160, True,   'hardswish', 2],
                [5, 960,  160, True,   'hardswish', 1],
                [5, 960,  160, True,   'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp,  c,   se,     nl,        s
                [3, 16,   16,  True,   'relu',    2],
                [3, 72,   24,  False,  'relu',    2],
                [3, 88,   24,  False,  'relu',    1],
                [5, 96,   40,  True,   'hardswish', 2],
                [5, 240,  40,  True,   'hardswish', 1],
                [5, 240,  40,  True,   'hardswish', 1],
                [5, 120,  48,  True,   'hardswish', 1],
                [5, 144,  48,  True,   'hardswish', 1],
                [5, 288,  96,  True,   'hardswish', 2],
                [5, 576,  96,  True,   'hardswish', 1],
                [5, 576,  96,  True,   'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name + "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish')

        # 初始化列表
        self.stages = nn.Sequential() # PyTorch 推荐直接使用 Sequential 或 ModuleList
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == 'large' else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                # 将 block_list 转为 Sequential 放入 stages
                self.stages.add_module(f"stage_{len(self.stages)}", nn.Sequential(*block_list))
                block_list = []
            
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        
        # 处理最后剩余的 blocks
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                if_act=True,
                act='hardswish'))
        
        # 放入最后一个 stage
        self.stages.add_module(f"stage_{len(self.stages)}", nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


# 辅助函数：make_divisible (保持不变，纯 Python 逻辑)
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v