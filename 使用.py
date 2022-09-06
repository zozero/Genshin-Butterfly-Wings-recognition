"""
用来找到晶蝶
先用帧差法找到正在运动的物体
然后使用已训练好的模型来判断是不是晶蝶
"""
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from 数据处理类 import 测试时数据处理
from 晶蝶模型类 import 晶蝶模型

模型存储路径 = '已训练的模型/晶蝶模型.ckpt3_8'


def 评估(图片):
    设备 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    模型 = 晶蝶模型()
    模型.to(设备)
    训练用数据 = 测试时数据处理(图片)
    测试用数据加载器 = DataLoader(训练用数据)
    模型.load_state_dict(torch.load(模型存储路径))
    模型.eval()
    for 图片 in 测试用数据加载器:
        图片 = 图片.to(设备)
        验证 = 模型(图片)
        print('验证结果：', 验证.item())
        if 0.8 < 验证.item() < 1.2:
            return True
        else:
            return False


if __name__ == '__main__':
    视频 = cv2.VideoCapture('视频/test4.mkv')

    while True:
        返回值, 帧 = 视频.read()
        返回值2, 帧2 = 视频.read()
        if 返回值 is False or 返回值2 is False:
            break

        灰色帧 = cv2.cvtColor(帧, cv2.COLOR_BGR2GRAY)
        灰色帧2 = cv2.cvtColor(帧2, cv2.COLOR_BGR2GRAY)

        偏差 = cv2.absdiff(灰色帧, 灰色帧2)
        # 二值化
        _, 偏差 = cv2.threshold(偏差, 10, 255, cv2.THRESH_BINARY)
        # cv2.imshow('piancha2', 偏差)
        # 膨胀操作
        核心 = np.ones((3, 3), np.uint8)
        偏差 = cv2.dilate(偏差, kernel=核心, iterations=5)
        # 腐蚀操作
        核心 = np.ones((5, 5), np.uint8)
        偏差 = cv2.erode(偏差, kernel=核心, iterations=1)
        # cv2.imshow('piancha', 偏差)
        # cv2.waitKey()
        # exit()

        # 找到轮廓
        轮廓列表, 阶层 = cv2.findContours(偏差, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        帧2拷贝 = 帧2.copy()
        # 绘制轮廓
        for 轮廓 in 轮廓列表:
            周长 = cv2.arcLength(轮廓, True)
            if 周长 > 300:
                x, y, 宽, 高 = cv2.boundingRect(轮廓)
                外扩值 = 20
                图片 = Image.fromarray(帧2)
                图片 = 图片.crop((x - 外扩值, y - 外扩值, x + 宽 + 外扩值, y + 高 + 外扩值))
                图片 = 图片.resize((50, 50))

                图片 = np.asarray(图片)
                # 显示已处理后的需要预测的图片
                cv2.imshow('tupian1', 图片)
                cv2.resizeWindow('tupian1', 230, 50)
                返回值 = 评估(图片)
                if 返回值:
                    # 显示预测后为晶蝶的图片
                    cv2.imshow('tupian2', 图片)
                    cv2.resizeWindow('tupian2', 230, 50)
                    # 经过评估是晶蝶的框显示为绿色
                    cv2.rectangle(帧2拷贝, (x - 外扩值, y - 外扩值), (x + 宽 + 外扩值, y + 高 + 外扩值), (0, 255, 0), 2)
                # 所有用帧差法找到的框显示为红色
                cv2.rectangle(帧2拷贝, (x - 外扩值 - 10, y - 外扩值 - 10), (x + 宽 + 外扩值 + 10, y + 高 + 外扩值 + 10), (0, 0, 255), 2)

        # 显示
        # 二值化，帧差法图片
        # cv2.imshow('piancha', 偏差)
        # 实际完整显示的效果
        cv2.imshow('zhen2', 帧2拷贝)

        键 = cv2.waitKey(5) & 0xff
        if 键 == 27:
            break
    视频.release()
    cv2.destroyAllWindows()
