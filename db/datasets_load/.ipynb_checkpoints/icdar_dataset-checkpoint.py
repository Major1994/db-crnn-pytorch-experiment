import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A  # 推荐使用 albumentations 进行数据增强

class ICDAR2015Dataset(Dataset):
    def __init__(self, img_dir, gt_dir, is_training=True, img_size=640):
        """
        Args:
            img_dir (str): 图片文件夹路径 (例如 'ICDAR2015/train')
            gt_dir (str): 标注文件夹路径 (例如 'ICDAR2015/train_gt')
            is_training (bool): 是否用于训练（决定是否进行数据增强）
            img_size (int): 图像缩放尺寸
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.is_training = is_training
        self.img_size = img_size
        
        # 获取所有图片路径
        self.image_list = [os.path.join(img_dir, img_name) 
                           for img_name in os.listdir(img_dir) 
                           if img_name.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # # 简单的数据增强配置 (可根据需要修改)
        # if is_training:
        #     self.transform = A.Compose([
        #         A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
        #         A.HorizontalFlip(p=0.5),
        #         A.ColorJitter(p=0.5),
        #         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        #     ])
        # else:
        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.image_list)

    def load_gt(self, gt_path):
        """
        读取标注文件，返回文本框坐标和标签
        ICDAR2015 格式: x1,y1,x2,y2,x3,y3,x4,y4,category
        """
        boxes = []
        texts = []
        
        if not os.path.exists(gt_path):
            return np.array(boxes, dtype=np.float32), texts

        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip().split(',')
            # 提取前8个数字作为坐标
            box_coords = [float(x) for x in line[:8]]
            # 提取标签 (第9个字段，可能是 '###" 或文字内容)
            label = line[8] if len(line) > 8 else "###"
            
            # 过滤掉 "Don't care" 区域 (通常标记为 ###)
            if label == "###":
                continue
                
            boxes.append(box_coords)
            texts.append(label)
            
        return np.array(boxes, dtype=np.float32), texts

    def __getitem__(self, idx):
        # 1. 读取图片
        img_path = self.image_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
        
        # 2. 读取标注
        gt_filename = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt' # 假设文件名对应
        # 注意：ICDAR2015 标注文件名通常带有 'gt_' 前缀，需根据实际情况调整
        if not os.path.exists(os.path.join(self.gt_dir, gt_filename)):
             gt_filename = 'gt_' + gt_filename # 尝试加上前缀
        
        gt_path = os.path.join(self.gt_dir, gt_filename)
        boxes, texts = self.load_gt(gt_path)
        
        # 3. 数据增强与变换
        # albumentations 需要传入 targets
        augmented = self.transform(image=image, bboxes=boxes.reshape(-1, 4, 2) if len(boxes) > 0 else [])
        
        image = augmented['image']
        # 还原 boxes 形状 (N, 4, 2) -> (N, 8)
        if len(boxes) > 0:
            boxes = augmented['bboxes']
            boxes = np.array(boxes).reshape(-1, 8)
        
        # 4. 转换为 Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() # HWC -> CHW
        
        # 5. 构造返回字典
        sample = {
            'image': image,
            'boxes': torch.from_numpy(boxes).float() if len(boxes) > 0 else torch.zeros(0, 8),
            'texts': texts,
            'shape': torch.tensor(image.shape[1:]) # 原始尺寸 (H, W)
        }
        
        return sample

# --- 测试代码 ---
if __name__ == "__main__":
    # 配置路径 (请修改为你自己的路径)
    IMG_DIR = '/home/meijian/db-crnn/db/icdar2015/text_localization/icdar_c4_train_imgs'
    GT_DIR = "/home/meijian/db-crnn/db/icdar2015/text_localization/test_icdar2015_label.txt"
    
    # 实例化数据集
    dataset = ICDAR2015Dataset(img_dir=IMG_DIR, gt_dir=GT_DIR, is_training=True)
    
    # 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # 模拟读取一个批次
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Image shape: {batch['image'].shape}")   # [Batch, 3, H, W]
        print(f"  Boxes shape: {batch['boxes'].shape}")   # [Batch, Max_Boxes, 8]
        print(f"  Texts: {batch['texts']}")
        
        # 这里可以接入你的 DBNet 模型
        # outputs = model(batch['image'])
        
        if i == 0: # 只测试一个批次
            break