"""
消除背景，找到前景
使用混合高斯模型
采取其他方法，不使用该方法
"""
import cv2

if __name__ == '__main__':
    视频 = cv2.VideoCapture('视频/test2.mkv')
    内核 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    图像背景 = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

    while True:
        返回值, 帧 = 视频.read()
        if 返回值 is False:
            break
        图像掩码 = 图像背景.apply(帧)
        # 形态学开运算去噪点
        图像掩码 = cv2.morphologyEx(图像掩码, cv2.MORPH_OPEN, kernel=内核)
        # 寻找轮廓
        轮廓列表, 阶层 = cv2.findContours(图像掩码, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for 轮廓 in 轮廓列表:
            周长 = cv2.arcLength(轮廓, True)
            if 周长 > 220:
                x, y, 宽, 高 = cv2.boundingRect(轮廓)
                cv2.rectangle(帧, (x, y), (x + 宽, y + 高), (0, 255, 0), 2)

        cv2.imshow('zhen', 帧)
        cv2.imshow('yanma', 图像掩码)
        键 = cv2.waitKey(10) & 0xff
        if 键 == 27:
            break
    视频.release()
    cv2.destroyAllWindows()
