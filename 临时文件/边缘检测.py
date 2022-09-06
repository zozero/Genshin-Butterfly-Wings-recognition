"""
使用康迪边缘检测（canny）
但和我预想的效果有差距......
"""
import cv2
import numpy as np

if __name__ == '__main__':
    视频 = cv2.VideoCapture('视频/test2.mkv')

    while True:
        返回值, 帧 = 视频.read()
        if 返回值 is False:
            break
        灰色帧 = cv2.cvtColor(帧, cv2.IMREAD_GRAYSCALE)
        值1 = cv2.Canny(灰色帧, 100, 150)

        轮廓列表, 阶层 = cv2.findContours(值1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for 轮廓 in 轮廓列表:
            周长 = cv2.arcLength(轮廓, True)
            if 周长 > 100:
                x, y, 宽, 高 = cv2.boundingRect(轮廓)
                cv2.rectangle(帧, (x, y), (x + 宽, y + 高), (0, 255, 0), 2)

        cv2.imshow('zhi', 值1)
        cv2.imshow('zhen', 帧)

        键 = cv2.waitKey(10) & 0xff
        if 键 == 27:
            break
    视频.release()
    cv2.destroyAllWindows()
