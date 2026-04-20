import torch

from neck.fpn import DBFPN
from heads.det_db_head import DBHead 
# 这里使用 torchvision 的 MobileNetV3 作为示例，或者你需要导入自定义的 MobileNetV3
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# 2. 计算 DBFPN 网络输出结果
# Paddle: paddle.randn -> PyTorch: torch.randn
# 注意：PyTorch 默认通道在前 (N, C, H, W)，与 Paddle 一致
fake_inputs = torch.randn(1, 3, 640, 640, dtype=torch.float32)

# 3. 初始化 Backbone 网络
# PyTorch 的 MobileNetV3 返回特征的方式可能需要微调，这里以标准 torchvision 模型为例
# 如果需要完全复现 PaddleOCR 的 MobileNetV3，请替换为你自定义的类
model_backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

# 获取输出通道数
# PaddleOCR 的 MobileNetV3 通常输出 [C2, C3, C4, C5] 四个层级
# torchvision 的模型结构略有不同，这里假设你有一个能返回多尺度特征的 Backbone
# 如果是自定义的 PaddleOCR 移植版，直接使用 model_backbone.out_channels
# 这里为了演示，手动指定常见的 MobileNetV3 输出通道配置 (需根据你的具体 Backbone 调整)
in_channels = [24, 40, 112, 960] # 示例通道数：对应 MobileNetV3 的不同阶段输出

# 4. 声明 FPN 网络
model_fpn = DBFPN(in_channels=in_channels, out_channels=256)

# 5. 模拟 Backbone 输出
# 注意：标准的 torchvision 模型只返回最后的特征图。
# 若要复现 PaddleOCR 的行为，你需要一个能返回中间层特征的 Backbone 实现。
# 这里假设 model_backbone 已经被修改为返回一个包含 4 个张量的列表。
# 如果直接运行这行代码会报错，因为 torchvision 模型返回的是单个 Tensor。
# 下面的代码仅为逻辑示意，实际需配合自定义 Backbone。

# 模拟 Backbone 输出 (因为标准 torchvision 模型不直接返回 list)
# 实际使用时，请确保你的 Backbone 返回的是 [c2, c3, c4, c5] 列表
try:
    # 尝试获取真实输出（如果你的 Backbone 已适配）
    # outs = model_backbone(fake_inputs) 
    
    # 这里为了代码能跑通，我们手动模拟 Backbone 输出的形状
    # 假设 c2, c3, c4, c5 的分辨率分别是输入的 1/2, 1/4, 1/8, 1/16 (或类似比例)
    c2 = torch.randn(1, 24, 320, 320)
    c3 = torch.randn(1, 40, 160, 160)
    c4 = torch.randn(1, 112, 80, 80)
    c5 = torch.randn(1, 960, 40, 40)
    outs = [c2, c3, c4, c5]
    
    # 6. 计算 FPN 输出
    fpn_outs = model_fpn(outs)

    # 7. 声明 Head 网络
    model_db_head = DBHead(in_channels=256)

    # 8. 打印 DBHead 网络
    print(model_db_head)

    # 9. 计算 Head 网络的输出
    # 注意：这里需要传入 training=False 或者确保模型处于 eval 模式，否则 DBHead 会尝试计算 threshold_maps
    model_db_head.eval() # 设置为评估模式，只返回 shrink_maps
    with torch.no_grad(): # 推理时不需要计算梯度
        db_head_outs = model_db_head(fpn_outs)

    # 10. 打印输出形状
    print(f"The shape of fpn outs: {fpn_outs.shape}")
    print(f"The shape of DB head outs: {db_head_outs['maps'].shape}")
    
except Exception as e:
    print(f"Error running backbone: {e}")
    print("Note: Ensure your Backbone returns a list of 4 tensors corresponding to C2, C3, C4, C5.")