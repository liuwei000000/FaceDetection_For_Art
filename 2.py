#coding=utf8
import cv2

# 一般笔记本的默认摄像头都是0
capInput = cv2.VideoCapture(0)
# 我们可以用这条命令检测摄像头是否可以读取数据
if not capInput.isOpened(): print('Capture failed because of camera')