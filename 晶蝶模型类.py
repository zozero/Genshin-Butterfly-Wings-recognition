import torch.nn
from torch import nn


def 初始化权重(模型):
    类名 = 模型.__class__.__name__

    if 类名.find('Conv') != -1:
        模型.weight.data.normal_(0.0, 0.02)
    elif 类名.find('BatchNorm') != -1:
        模型.weight.data.normal_(1.0, 0.02)
        模型.bias.data.fill_(0)


class 晶蝶模型(nn.Module):
    def __init__(self):
        super(晶蝶模型, self).__init__()
        self.卷积1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.线性整流函数1 = nn.ReLU()
        self.池化1 = nn.MaxPool2d(2, 2)

        self.全连接层1 = nn.Linear(4 * 25 * 25, 2 * 25)
        self.全连接层2 = nn.Linear(2 * 25, 2)
        self.全连接层3 = nn.Linear(2, 1)

        self.线性整流函数2 = nn.ReLU()

    def forward(self, 输入):
        输出 = self.卷积1(输入)
        输出 = self.线性整流函数1(输出)
        输出 = self.池化1(输出)
        # print('1：', 输出.size())

        _, T, b, h = tuple(输出.size())
        输出 = 输出.view(T * b * h)

        输出 = self.线性整流函数2(self.全连接层1(输出))
        输出 = self.线性整流函数2(self.全连接层2(输出))
        输出 = self.全连接层3(输出)

        return 输出


# 直接使用预设的函数了，所以以下代码放弃。
class 晶蝶损失(nn.Module):
    def __init__(self):
        super(晶蝶损失, self).__init__()

    def forward(self, 输入, 目标):
        """
        让是晶蝶的趋近于1，不是的趋近于0
        :param 输入:
        :param 目标:
        :return:
        """
        return (输入 - 目标)**2
