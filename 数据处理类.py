import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def 载入图片(图片路径):
    图片 = Image.open(图片路径)
    图片 = 图片.resize((50, 50))
    图片 = np.asarray(图片)
    图片 = cv2.cvtColor(图片, cv2.COLOR_RGB2BGR)
    # cv2.imshow('zairu', 图片)
    # cv2.waitKey()
    # exit()
    return 图片


class 数据处理(Dataset):
    def __init__(self, 图片目录='图片/训练'):
        self.转张量 = transforms.ToTensor()
        self.图片列表 = []
        self.图片名列表 = os.listdir(图片目录)
        for 图片名 in self.图片名列表:
            图片 = 载入图片(os.path.join(图片目录, 图片名))
            图片 = self.转张量(图片) * 10
            # torch.sub()
            # 图片.sub_(0.5).div_(0.5)
            # 图片 = 图片.unsqueeze(0)
            self.图片列表.append(图片)
        # print(len(self.图片列表))

    def __len__(self):
        return len(self.图片列表)

    def __getitem__(self, 索引):
        return self.图片名列表[索引], self.图片列表[索引]


class 测试时数据处理(Dataset):
    def __init__(self, 图片):
        self.转张量 = transforms.ToTensor()
        图片 = self.转张量(图片) * 10
        # 图片.sub_(0.5).div_(0.5)
        self.图片列表 = [图片]

    def __len__(self):
        return len(self.图片列表)

    def __getitem__(self, 索引):
        return self.图片列表[索引]
