"""
使用目标追踪的方式
用来采集晶蝶图片数据
"""
from PIL import Image

import cv2

if __name__ == '__main__':
    多个追踪器 = cv2.legacy.MultiTracker_create()

    捕获的视频 = cv2.VideoCapture('视频/test4.mkv')
    次数 = 1
    while True:
        帧 = 捕获的视频.read()
        一帧 = 帧[1]
        if 一帧 is None:
            break
        (多个成功, 多个盒子) = 多个追踪器.update(一帧)
        for 盒子 in 多个盒子:
            (x, y, 宽, 高) = [int(v) for v in 盒子]
            cv2.rectangle(一帧, (x - 2, y - 2), (x + 宽 + 2, y + 高 + 2), (0, 255, 0), 2)
            if x != 0 and y != 0:
                图片 = cv2.cvtColor(一帧, cv2.COLOR_BGR2RGB)
                图片 = Image.fromarray(图片)
                图片 = 图片.crop((x, y, x + 宽, y + 高))
                图片.save('../图片/' + str(次数) + '.jpg')
                次数 += 1

        cv2.imshow("zhen", 一帧)
        键 = cv2.waitKey(10) & 0xFF
        if 键 == ord("s"):
            盒子 = cv2.selectROI("zhen", 一帧, fromCenter=False, showCrosshair=True)
            追踪器 = cv2.legacy.TrackerKCF_create()
            多个追踪器.add(追踪器, 一帧, 盒子)

        elif 键 == 27:
            break

    捕获的视频.release()
    cv2.destroyAllWindows()
