import random
import time
from PIL import Image
import hashlib
import numpy as np
from PIL import ImageFile
from pdb import set_trace as stop
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_unk_mask_indices(image,testing,num_labels,known_labels,epoch=1):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array
        # 哈希种子
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        # # 在所有标签内选择 “未知标签个数” 长度的数组
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        # 有监督学习，图像有标签
        if known_labels>0:
            random.seed()
            # 随机选择 0-75% 的标签作为已知标签
            # 如何保证每张图像或者大部分图像均有 known_label ？ 
            # 图像的平均标签数 ？
            num_known = random.randint(0,int(num_labels*0.75))
        # 无监督学习
        else:
            num_known = 0
        # 在 range(num_labels) 中选择 (num_labels-num_known) 长度的数组
        # 在所有标签内选择 “未知标签个数” 长度的数组
        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))
    return unk_mask_indices

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def image_loader(path,transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')

    if transform is not None:
        image = transform(image)

    return image
