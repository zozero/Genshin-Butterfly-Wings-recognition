"""
截取是视频中第一帧，将非晶蝶的部分截取出来，用于后续训练，这些截取的图片标签为0
"""
import time

from PIL import Image

import cv2

if __name__ == '__main__':
    捕获的视频 = cv2.VideoCapture('视频/test4.mkv')
    帧 = 捕获的视频.read()
    一帧 = 帧[1]
    一帧 = cv2.cvtColor(一帧, cv2.COLOR_BGR2RGB)
    图片 = Image.fromarray(一帧)
    # 图片.show()
    print(图片.size)
    当前时间 = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print(当前时间)
    基数 = 300
    列数 = int(图片.size[0] / 基数)
    行数 = int(图片.size[1] / 基数)
    for 列 in range(列数):
        for 行 in range(行数):
            截取的图片 = 图片.crop((基数 * 列, 基数 * 行, 基数 * 列 + 基数, 基数 * 行 + 基数))
            截取的图片.save(f'非晶蝶/{str(基数)}_{str(列)}_{str(行)}_{当前时间}.jpg')
            print(f"行：{列}，列：{行}")

    捕获的视频.release()
