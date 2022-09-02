import os.path
from PIL import Image

import cv2 as cv

if __name__ == '__main__':
    多个追踪器 = cv.legacy.MultiTracker_create()

    捕获的视频 = cv.VideoCapture('视频/test2.mp4')
    次数 = 1
    while True:
        帧 = 捕获的视频.read()
        一帧 = 帧[1]
        if 一帧 is None:
            break
        (多个成功, 多个盒子) = 多个追踪器.update(一帧)
        for 盒子 in 多个盒子:
            (x, y, 宽, 高) = [int(v) for v in 盒子]
            cv.rectangle(一帧, (x - 2, y - 2), (x + 宽 + 2, y + 高 + 2), (0, 255, 0), 2)
            if x != 0 and y != 0:
                图片 = cv.cvtColor(一帧, cv.COLOR_BGR2RGB)
                图片 = Image.fromarray(图片)
                图片 = 图片.crop((x, y, x + 宽, y + 高))
                图片.save('图片/' + str(次数) + '.jpg')

        cv.imshow("zhen", 一帧)
        键 = cv.waitKey(100) & 0xFF
        if 键 == ord("s"):
            盒子 = cv.selectROI("zhen", 一帧, fromCenter=False, showCrosshair=True)
            追踪器 = cv.legacy.TrackerKCF_create()
            多个追踪器.add(追踪器, 一帧, 盒子)
        elif 键 == 27:
            break
        次数 += 1
    捕获的视频.release()
    cv.destroyAllWindows()
