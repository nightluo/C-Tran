import random
import time
from PIL import Image
import hashlib
import numpy as np
from PIL import ImageFile
from pdb import set_trace as stop
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_unk_mask_indices(image, testing, num_labels, known_labels, epoch=1):
    # testing: 测试时为 True，训练时为 False
    # 标签种类 num_labels = 80
    # 使用 lmt 时 args.train_known_labels = 100，否则默认为 0 
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array
        # 哈希种子
        # hashlib.sha1().hexdigest()，哈希算法，hashlib.sha1() 创建加密实例，hexdigest() 返回摘要，作为十六进制数据字符串值
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        # 在所有标签内选择 “未知标签个数” 长度的数组，获得要 mask 的 label_idx
        unk_mask_indices = random.sample(range(num_labels), (num_labels - int(known_labels)))
        """
        1、将输入的要测试图像从 PIL Image 转换为 array
        2、对 array 图像使用哈希算法, 生成不可逆混杂字符串
        3、使用哈希种子初始化 random
        4、随机抽取未知标签个数的需要 mask 的 label_idx
        """
    else:
        # sample random number of known labels during training
        if known_labels > 0:
            # 有监督学习，图像有标签
            # 使用 lmt 训练方法
            random.seed()
            # 随机选择 0-75% 的标签作为已知标签
            # 如何保证每张图像或者大部分图像均有 known_label ？ 
            # 图像的平均标签数 ？
            num_known = random.randint(0, int(num_labels * 0.75))
            # print(f"num_known = {num_known}")
        
        else:
            # 无监督学习
            # 不使用 lmt 训练方法
            num_known = 0

        # 在 range(num_labels) 中选择 (num_labels - num_known) 长度的数组
        # 在所有标签内选择 “未知标签个数” 长度的数组，获得要 mask 的 label_idx
        unk_mask_indices = random.sample(range(num_labels), (num_labels - num_known))
    
    # print(f"unk_mask_indices = {unk_mask_indices}")
        
    return unk_mask_indices

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def image_loader(path, transform):
    try:
        # 直接打开图像文件
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')

    if transform is not None:
        image = transform(image)

    return image
