"""
计算前后帧的差别，锁定晶蝶的可能位置
"""
import cv2
import numpy as np

if __name__ == '__main__':
    视频 = cv2.VideoCapture('视频/test3.mkv')

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
        核心 = np.ones((7, 7), np.uint8)
        偏差 = cv2.erode(偏差, kernel=核心, iterations=3)
        # cv2.imshow('piancha', 偏差)
        # cv2.waitKey()
        # exit()

        # 找到轮廓
        轮廓列表, 阶层 = cv2.findContours(偏差, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        for 轮廓 in 轮廓列表:
            周长 = cv2.arcLength(轮廓, True)
            外扩值 = 30
            if 周长 > 200:
                x, y, 宽, 高 = cv2.boundingRect(轮廓)
                cv2.rectangle(帧2, (x, y), (x + 宽 + 外扩值, y + 高 + 外扩值), (0, 255, 0), 2)

        # 显示
        cv2.imshow('piancha', 偏差)
        cv2.imshow('zhen', 帧2)

        键 = cv2.waitKey(10) & 0xff
        if 键 == 27:
            break
    视频.release()
    cv2.destroyAllWindows()
