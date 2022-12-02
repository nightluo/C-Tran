import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
from dataloaders.data_utils import get_unk_mask_indices, image_loader

class Coco80Dataset(Dataset):
    def __init__(self, split, num_labels, data_file, img_root, annotation_dir, max_samples=-1, transform=None, known_labels=0, testing=False, analyze=False):
        # data_file = os.path.join(coco_root,“train.data"), train.data 文件
        self.split=split
        self.split_data = pickle.load(open(data_file,'rb'))
        
        # max_samples 默认为 -1
        if max_samples != -1:
            self.split_data = self.split_data[0:max_samples]

        self.img_root = img_root
        # 数据增强操作：裁剪、放缩、标准化
        self.transform = transform
        # 标签种类 num_labels = 80
        self.num_labels = num_labels
        # 使用 lmt 时 args.train_known_labels = 100，否则默认为 0 
        self.known_labels = known_labels
        self.testing = testing
        self.epoch = 1

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        """
        split_data[32320]:{
            'image_id': '48772', 
            'file_name': 'COCO_train2014_000000048772.jpg', 
            'objects': [
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'caption': 'A man and a woman sitting down behind a table with bananas on it.'
        }
        """
        # print(f"split_data[{idx}]:{self.split_data[idx]}")

        # 获取 batch_size 中索引为 idx 的对象的图像 id
        image_ID = self.split_data[idx]['file_name']
        # 图像路径 + 图像id
        img_name = os.path.join(self.img_root,image_ID)
        # 根据路径获取图像并进行对应的图像增强操作
        image = image_loader(img_name, self.transform)
        # 0-1 编码的标签信息
        labels = self.split_data[idx]['objects']
        labels = torch.Tensor(labels)
        
        # 标签种类 num_labels = 80
        # 使用 lmt 时 args.train_known_labels = 100，否则默认为 0 
        unk_mask_indices = get_unk_mask_indices(image, self.testing, self.num_labels, self.known_labels)
        
        mask = labels.clone()
        # 根据索引值，在指定位置生成 mask
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = image_ID
        return sample


