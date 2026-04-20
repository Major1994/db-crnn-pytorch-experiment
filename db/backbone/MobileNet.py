import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3MultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 1. 加载官方模型
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        
        # 2. 获取特征提取部分 (去掉最后的分类器和池化层)
        self.features = model.features
        # print(self.features)
        
        # 3. 定义输出层索引
        # MobileNetV3-Large 的默认结构大约有 16-19 层 (取决于具体版本)
        # 我们需要根据下采样倍率选择层索引。
        # 假设输入 640x640:
        # - 索引 2 (stride=2): 320x320 (通常作为浅层特征)
        # - 索引 4 (stride=4): 160x160
        # - 索引 7 (stride=8):  80x80
        # - 索引 12 (stride=16): 40x40
        # - 索引 -1 (stride=32): 20x20 (最深层特征)
        
        # 这里我们选择 stride=8, 16, 32 的层作为典型的多尺度输出
        self.out_indices = (3,6,12,16) 

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        
        return outputs

# --- 测试代码 ---
model = MobileNetV3MultiScale(pretrained=True)
model.eval()

fake_input = torch.randn(1, 3, 640, 640)
outs = model(fake_input)

print(f"输入形状: {fake_input.shape}")
print(f"输出了 {len(outs)} 层特征图：")
for i, out in enumerate(outs):
    print(f"层级 {i} :  {out.shape}")