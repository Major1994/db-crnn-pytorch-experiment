import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2 # 用于直接转为 Tensor
import json
import os
import numpy as np

def custom_collate_fn(batch):
    """
    Collate 函数保持不变，但要注意 boxes 现在已经是 Tensor 了
    """
    images = []
    boxes_list = []
    texts_list = []
    paths = []

    for item in batch:
        images.append(item['image'])
        # Albumentations 处理后，boxes 已经是 list of tensors 或 list of lists
        if isinstance(item['boxes'], np.ndarray):
            boxes_list.append(torch.from_numpy(item['boxes']))
        else:
            boxes_list.append(item['boxes'])
            
        texts_list.append(item['texts'])
        paths.append(item['path'])

    images = torch.stack(images, dim=0) 

    return {
        'image': images,     
        'boxes': boxes_list, 
        'texts': texts_list, 
        'path': paths        
    }

class ICDAR2015_JsonDataset(Dataset):
    def __init__(self, img_dir, label_file, is_training=True, img_size=640):
        super(ICDAR2015_JsonDataset, self).__init__()
        self.img_dir = img_dir
        self.label_file = label_file
        self.is_training = is_training
        self.img_size = 640
        
        self.data_lines = self.get_image_info_list(label_file)
        import pdb
        pdb.set_trace()
        # --- 关键修改：定义 Albumentations 变换 ---
        if is_training:
            self.transform = A.Compose([
                # 1. 几何变换：同时作用于图像和 bbox
                # A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=0.8),
                # A.HorizontalFlip(p=0.5),
                
                # 2. 光度变换：只作用于图像
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),
                
                # 3. 归一化并转为 Tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], 
            # 指定 bbox 格式为 COCO 风格: [x_min, y_min, width, height]
            # 注意：Albumentations 内部计算通常使用这种格式，或者 'pascal_voc' ([x_min, y_min, x_max, y_max])
            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'])
            )
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def get_image_info_list(self, file_path):
        data_lines = []
        with open(file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n")
                if line:
                    data_lines.append(line)
        return data_lines

    def __getitem__(self, idx):
        data_line = self.data_lines[idx]
        
        try:
            # 1. 解析 JSON
            img_path_part, json_str = data_line.split('\t', 1)
            annotations = json.loads(json_str)
            
            img_path = os.path.join(self.img_dir, img_path_part)
            
            if not os.path.exists(img_path):
                raise Exception(f"{img_path} does not exist!")
            
            # 2. 读取图片 (Albumentations 直接读取 numpy 数组)
            image = np.array(Image.open(img_path).convert('RGB'))
            orig_h, orig_w = image.shape[:2]
            
            import pdb
            pdb.set_trace()
            # 3. 准备 Albumentations 需要的数据
            # Albumentations 需要 bboxes 列表和对应的 labels 列表
            bboxes = [] 
            class_labels = [] # 即使我们不用类别，也必须提供这个列表来对应每个框
            
            for ann in annotations:
                transcription = ann.get('transcription', '')
                points = ann.get('points', [])
                
                if transcription == "###":
                    continue
                
                if len(points) == 4:
                    # --- 坐标格式转换 ---
                    # 原始数据: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    # 目标格式 (COCO): [x_min, y_min, width, height]
                    
                    x_coords = [p for p in points]
                    y_coords = [p for p in points]
                    
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # 过滤掉极小的框（可选，防止增强后出错）
                    if width > 0 and height > 0:
                        bboxes.append([x_min, y_min, width, height])
                        class_labels.append(transcription) # 暂时把文本当 label 存
            
            # 4. 应用变换
            # 注意：这里 Albumentations 会自动计算新的 bbox 坐标
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            image = transformed['image'] # 已经是 Tensor [C, H, W]
            transformed_bboxes = transformed['bboxes'] # 列表，格式仍为 COCO [x_min, y_min, w, h]
            texts = transformed['class_labels'] # 对应的文本列表
            
            # 5. 坐标格式还原
            # 模型需要: [x1, y1, x2, y2, x3, y3, x4, y4] (8个坐标)
            # 目前 transformed_bboxes 是 [x_min, y_min, w, h]
            
            final_boxes = []
            for box in transformed_bboxes:
                x_min, y_min, w, h = box
                x_max = x_min + w
                y_max = y_min + h
                
                # 这里我们只能还原成矩形框。
                # 如果你需要严格的四边形，Albumentations 支持 'polygon' 格式，
                # 但 DBNet 的 GT 生成通常也是基于矩形或外接矩形的，所以这里用矩形通常没问题。
                # 展平为 8 维向量
                final_boxes.append([x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max])
            
            final_boxes = np.array(final_boxes, dtype=np.float32)

            data = {
                'image': image, 
                'boxes': final_boxes, # Numpy array [N, 8]
                'texts': texts, 
                'path': img_path
            }
            return data
            
        except Exception as e:
            print(f"Error loading {data_line}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self.data_lines))

    def __len__(self):
        return len(self.data_lines)

label_path = "icdar2015/text_localization/train_icdar2015_label.txt"
img_dir = "icdar2015/text_localization/" 
train_dataset = ICDAR2015_JsonDataset(img_dir=img_dir, label_file=label_path, is_training=True)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True
)
print(train_dataset)
print(next(iter(train_loader)))
