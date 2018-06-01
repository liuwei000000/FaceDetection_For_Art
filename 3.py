#!Anaconda/anaconda/python
#coding: utf-8

"""
从视屏中识别人脸，并实时标出面部特征点
"""

import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv
import time
from PIL import Image
import os

INTERVAL = 2   #秒间隔

'''
PIL.Image转换成OpenCV格式

image = Image.open("plane.jpg")  
image.show()  
img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)  
cv2.imshow("OpenCV",img)  

==============================
OpenCV转换成PIL.Image格式

img = cv2.imread("plane.jpg")  
cv2.imshow("OpenCV",img)  
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
image.show()
'''

def cfiles(dir):
    c = 0    #计数大文件夹下共有多少个小文件夹
    for filename in os.listdir(dir):
        c += 1
    return c

class face_emotion():

    def __init__(self):
        # 使用特征提取器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        #建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(0)
        # 设置视频参数，propId设置的视频参数，value设置的参数值
        # self.cap.set(3, 480)
        self.width = self.cap.get(3)
        self.height = self.cap.get(4)
        # 截图screenshoot的计数器
        self.cnt = cfiles('./ims')
        self.currenttime = time.time()

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()        

    def emotion(self, shape, im_rd, d):
        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []
        
        # 计算人脸热别框边长     
        face_width = d.right() - d.left()
                
        # 圆圈显示每个特征点
        for i in range(68):
            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, #圆大小
                (0, 255, 0), -1, 8)
            #cv2.putText(im_rd, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #            (255, 255, 255))
            # 分析任意n点的位置关系来作为表情识别的依据
            mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴咧开程度
            mouth_higth = (shape.part(66).y - shape.part(62).y) / face_width  # 嘴巴张开程度
            # print("嘴巴宽度与识别框宽度之比：",mouth_width_arv)
            # print("嘴巴高度与识别框高度之比：",mouth_higth_arv)

            # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
            brow_sum = 0  # 高度之和
            frown_sum = 0  # 两边眉毛距离之和
            for j in range(17, 21):
                brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                frown_sum += shape.part(j + 5).x - shape.part(j).x
                line_brow_x.append(shape.part(j).x)
                line_brow_y.append(shape.part(j).y)

            # self.brow_k, self.brow_d = self.fit_slr(line_brow_x, line_brow_y)  # 计算眉毛的倾斜程度
            tempx = np.array(line_brow_x)
            tempy = np.array(line_brow_y)
            z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
            self.brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的

            brow_hight = (brow_sum / 10) / face_width  # 眉毛高度占比
            brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比
            # print("眉毛高度与识别框高度之比：",round(brow_arv/self.face_width,3))
            # print("眉毛间距与识别框高度之比：",round(frown_arv/self.face_width,3))

            # 眼睛睁开程度
            eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                       shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
            eye_hight = (eye_sum / 4) / face_width
            # print("眼睛睁开距离与识别框高度之比：",round(eye_open/self.face_width,3))

            # 分情况讨论
            # 张嘴，可能是开心或者惊讶
            if round(mouth_higth >= 0.03):
                if eye_hight >= 0.056:
                    cv2.putText(im_rd, "amazing", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, 4)
                else:
                    cv2.putText(im_rd, "happy", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, 4)

            # 没有张嘴，可能是正常和生气
            else:
                if self.brow_k <= -0.3:
                    cv2.putText(im_rd, "angry", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, 4)
                else:
                    cv2.putText(im_rd, "nature", (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2, 4)

    def capture_face(self):
        bc_fm = 0
        bc_im_p = None
        bc_count = 0
        bc_faces = None
        # cap.isOpened（） 返回true/false 检查初始化是否成功
        while(self.cap.isOpened()):
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()
            if not flag:
                continue

            # 每帧数据延时1ms，延时为0读取的是静态帧
            k = cv2.waitKey(1)

            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            
            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
            faces = self.detector(img_gray, 0)

            # 待会要显示在屏幕上的字体
            font = cv2.FONT_HERSHEY_SIMPLEX

            # 如果检测到人脸
            if(len(faces)!=0):
                # 取清晰度
                t = self.variance_of_laplacian(img_gray)

                if (t > bc_fm ):
                    bc_fm = t
                    bc_im_p = im_rd.copy()
                    bc_faces = faces
                bc_count = bc_count + 1

                # 对每个人脸都标出68个特征点
                for i in range(len(faces)):
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(faces):
                        # 用红色矩形框出人脸
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))                   
                        # 使用预测器得到68点数据的坐标
                        # shape = self.predictor(im_rd, d)
                        # self.emotion(shape, im_rd, d)

                # 标出人脸数
                cv2.putText(im_rd, "Faces: "+str(len(faces)), (20,50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # 没有检测到人脸
                cv2.putText(im_rd, "No Face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # 添加说明
            #im_rd = cv2.putText(im_rd, "S: " + str(fm), (20, 400), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "Q: quit", (20, 450), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # 按下q键退出
            if(k == ord('q')):
                break

            # 窗口显示
            cv2.imshow("camera", im_rd)
            #cv2.imshow("camera_gray", img_gray)
            #print(time.time())

            if (self.currenttime // INTERVAL != time.time() // INTERVAL):
                if bc_fm == 0 or len(bc_faces) == 0:
                    continue
                bc_fm = 0
                self.currenttime = time.time()
                t = time.strftime('%y%m%d-%H%M%S',time.localtime())
                # 保存图片
                img = Image.fromarray(cv2.cvtColor(bc_im_p,cv2.COLOR_BGR2RGB))
                for i in range(len(bc_faces)):
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(bc_faces):
                        il = d.left()
                        if il < 0: il = 0                        
                        it = d.top()
                        if it < 0 : it = 0
                        ir = d.right()
                        if ir > self.width : ir = self.width
                        ib = d.bottom()
                        if ib > self.height : ib = self.height
                        print("%d, %d, %d, %d " % (il, it, ir, ib))
                        img_face = img.crop((il, it, ir, ib))
                        img_face.save('./ims/' + t + '-' + str(self.cnt)+".jpg")
                        self.cnt += 1
                img.save('./ims/Whole-' + t + '-' + str(self.cnt)+".jpg")
                print(t + ' | Filter:' + str(bc_count) + ' | No:' + str(self.cnt))
                self.cnt += 1
                bc_count = 0

        # 释放摄像头
        self.cap.release()
        # 删除建立的窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    my_face = face_emotion()
    my_face.capture_face()