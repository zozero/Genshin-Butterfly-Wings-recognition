import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from 数据处理类 import 数据处理
# from 晶蝶模型类 import 晶蝶损失
from 晶蝶模型类 import 晶蝶模型

# 随机种子 = 22
# torch.manual_seed(随机种子)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False  # 避免因为随机性产生出差异
# np.random.seed(随机种子)

模型存储路径 = '已训练的模型/晶蝶模型.ckpt3_8'


def 评估(模型, 设备):
    训练用数据 = 数据处理(图片目录='图片/测试')
    测试用数据加载器 = DataLoader(训练用数据)
    模型.load_state_dict(torch.load(模型存储路径))
    模型.eval()
    for 图片名, 图片 in 测试用数据加载器:
        图片 = 图片.to(设备)
        验证 = 模型(图片)
        print('图片名：', 图片名, '验证结果：', 验证)


if __name__ == '__main__':
    设备 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    模型 = 晶蝶模型()
    模型.to(设备)

    是否继续训练 = True
    if os.path.exists(模型存储路径) and 是否继续训练:
        模型.load_state_dict(torch.load(模型存储路径))
    else:
        # 模型.apply(初始化权重)
        pass

    # 损失函数 = 晶蝶损失()
    损失函数 = torch.nn.L1Loss()
    损失函数.to(设备)
    优化器 = optim.Adam(模型.parameters(), lr=0.00001)
    print(模型)
    目标值 = torch.tensor([1]).to(设备)

    训练用数据 = 数据处理()
    训练用数据加载器 = DataLoader(训练用数据, shuffle=True)
    计数 = 0
    for 轮回 in range(9):
        for _, 图片 in 训练用数据加载器:
            图片 = 图片.to(设备)
            模型.train()

            优化器.zero_grad()
            预测 = 模型(图片)
            损失值 = 损失函数(预测, 目标值)
            模型.zero_grad()
            损失值.backward()
            优化器.step()
            # print('预测值：', 预测.item())
            计数 += 1
            print(f'轮回：{轮回}，损失值：{损失值.item():.8f}，计数：{计数}')
            # exit()
    torch.save(模型.state_dict(), 模型存储路径)
    评估(模型, 设备)
    pass
