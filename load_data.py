import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from dataloaders.voc2007_20 import Voc07Dataset
from dataloaders.vg500_dataset import VGDataset
from dataloaders.coco80_dataset import Coco80Dataset
from dataloaders.news500_dataset import NewsDataset
from dataloaders.coco1000_dataset import Coco1000Dataset
from dataloaders.cub312_dataset import CUBDataset
import warnings
warnings.filterwarnings("ignore")


def get_data(args):
    dataset = args.dataset
    data_root=args.dataroot
    batch_size=args.batch_size
    workers=args.workers
    # 放缩，rescale = args.scale_size = 640
    rescale=args.scale_size
    # 随即裁剪，random_crop = args.crop_size = 576
    random_crop=args.crop_size

    # 聚类处理的 CUB, attr_group_dict
    attr_group_dict=args.attr_group_dict
    # 聚类处理的 CUB, groups for CUB test time intervention
    n_groups=args.n_groups

    # 基于 ImageNet 数据集的 mean 和 std 参数进行标准化
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    scale_size = rescale
    crop_size = random_crop

    # optimization 的 test_batch_size 默认为 -1
    if args.test_batch_size == -1:
        args.test_batch_size = batch_size
    
    # 放缩到统一的尺寸 scale_size 640
    # 随机裁剪再放缩到统一的裁剪尺寸 crop_size 576
    trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        # 随机选择一个裁剪尺寸进行裁剪
                                        transforms.RandomChoice([
                                        transforms.RandomCrop(640),
                                        transforms.RandomCrop(576),
                                        transforms.RandomCrop(512),
                                        transforms.RandomCrop(384),
                                        transforms.RandomCrop(320)
                                        ]),
                                        # 将裁剪的小块放缩成统一的 crop_size
                                        transforms.Resize((crop_size, crop_size)),
                                        # 翻转
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # 正则化
                                        normTransform])

    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                        # 不需要随机裁剪，直接中心裁剪 crop_size 尺寸的小块
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        normTransform])

    test_dataset = None
    test_loader = None
    # ？
    drop_last = False
    if dataset == 'coco':
        # 数据集根目录
        coco_root = os.path.join(data_root,'coco')
        # 每张图的文本信息目录
        ann_dir = os.path.join(coco_root,'annotations_pytorch')
        # 训练图像
        train_img_root = os.path.join(coco_root,'train2014')
        # 测试图像
        test_img_root = os.path.join(coco_root,'val2014')

        train_data_name = 'train.data'
        val_data_name = 'val_test.data'
        
        train_dataset = Coco80Dataset(
            # split 看起来并没有用处
            split='train',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root, train_data_name),
            img_root=train_img_root,
            annotation_dir=ann_dir,
            # 默认为 -1，设置最大样本数，why？
            max_samples=args.max_samples,
            transform=trainTransform,
            # 使用 lmt 方法时，args.train_known_labels=100
            known_labels=args.train_known_labels,
            testing=False)

        valid_dataset = Coco80Dataset(
            split='val',
            num_labels=args.num_labels,
            data_file=os.path.join(coco_root, val_data_name),
            img_root=test_img_root,
            annotation_dir=ann_dir,
            max_samples=args.max_samples,
            transform=testTransform,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'coco1000':
        ann_dir = os.path.join(data_root,'coco','annotations_pytorch')
        data_dir = os.path.join(data_root,'coco')
        train_img_root = os.path.join(data_dir,'train2014')
        test_img_root = os.path.join(data_dir,'val2014')
        train_dataset = Coco1000Dataset(ann_dir, data_dir, split = 'train', transform = trainTransform,known_labels=args.train_known_labels,testing=False)
        valid_dataset = Coco1000Dataset(ann_dir, data_dir, split = 'val', transform = testTransform,known_labels=args.test_known_labels,testing=True)
    
    elif dataset == 'vg':
        vg_root = os.path.join(data_root,'VG')
        train_dir=os.path.join(vg_root,'VG_100K')
        train_list=os.path.join(vg_root,'train_list_500.txt')
        test_dir=os.path.join(vg_root,'VG_100K')
        test_list=os.path.join(vg_root,'test_list_500.txt')
        train_label=os.path.join(vg_root,'vg_category_500_labels_index.json')
        test_label=os.path.join(vg_root,'vg_category_500_labels_index.json')

        train_dataset = VGDataset(
            train_dir,
            train_list,
            trainTransform, 
            train_label,
            known_labels=0,
            testing=False)
        valid_dataset = VGDataset(
            test_dir,
            test_list,
            testTransform,
            test_label,
            known_labels=args.test_known_labels,
            testing=True)
    
    elif dataset == 'news':
        drop_last=True
        ann_dir = '/bigtemp/jjl5sw/PartialMLC/data/bbc_data/'

        train_dataset = NewsDataset(ann_dir, split = 'train', transform = trainTransform,known_labels=0,testing=False)
        valid_dataset = NewsDataset(ann_dir, split = 'test', transform = testTransform,known_labels=args.test_known_labels,testing=True)
    
    elif dataset=='voc':
        voc_root = os.path.join(data_root,'voc/VOCdevkit/VOC2007/')
        img_dir = os.path.join(voc_root,'JPEGImages')
        anno_dir = os.path.join(voc_root,'Annotations')
        train_anno_path = os.path.join(voc_root,'ImageSets/Main/trainval.txt')
        test_anno_path = os.path.join(voc_root,'ImageSets/Main/test.txt')

        train_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=train_anno_path,
            image_transform=trainTransform,
            labels_path=anno_dir,
            known_labels=args.train_known_labels,
            testing=False,
            use_difficult=False)
        valid_dataset = Voc07Dataset(
            img_dir=img_dir,
            anno_path=test_anno_path,
            image_transform=testTransform,
            labels_path=anno_dir,
            known_labels=args.test_known_labels,
            testing=True)

    elif dataset == 'cub':
        drop_last=True
        resol=299
        resized_resol = int(resol * 256/224)
        
        trainTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])

        testTransform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
        
        cub_root = os.path.join(data_root,'CUB_200_2011')
        image_dir = os.path.join(cub_root,'images')
        train_list = os.path.join(cub_root,'class_attr_data_10','train_valid.pkl')
        valid_list = os.path.join(cub_root,'class_attr_data_10','train_valid.pkl')
        test_list = os.path.join(cub_root,'class_attr_data_10','test.pkl')

        train_dataset = CUBDataset(image_dir, train_list, trainTransform, known_labels=args.train_known_labels,attr_group_dict=attr_group_dict,testing=False,n_groups=n_groups)
        valid_dataset = CUBDataset(image_dir, valid_list, testTransform, known_labels=args.test_known_labels, attr_group_dict=attr_group_dict,testing=True,n_groups=n_groups)
        test_dataset = CUBDataset(image_dir, test_list, testTransform, known_labels=args.test_known_labels, attr_group_dict=attr_group_dict,testing=True, n_groups=n_groups)
        
    else:
        print('no dataset avail')
        exit(0)

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=drop_last, pin_memory=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader,valid_loader,test_loader
