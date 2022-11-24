from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
import sys
import cv2
from Ui_Mainwindow import Ui_MainWindow

import numpy as np
import queue, threading, time
from configparser import ConfigParser
from numpy.core.numeric import zeros_like
import matplotlib.pyplot as plt
import colorsys
import math
import random
from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import datetime
import os
import csv
import json
from mpl_toolkits.mplot3d import Axes3D


class MyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera_2 = QtCore.QTimer()
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.slot_init()  # 初始化槽函数
        self.count = 0

        #hsv transform to rgb format
        f = open('20220423-program/bosparameters.json', 'r')
        content = f.read()
        self.para = json.loads(content)
        self.flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        # 方案0大分辨率，方案1小分辨率
        self.solution = self.para['solution'][1]



    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open.clicked.connect(
            self.button_open_clicked)  # 若该按键被点击，则调用button_open_clicked()
        self.timer_camera.timeout.connect(
            self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(
            self.close
        )  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_getdisplacement.clicked.connect(
            self.button_get_displacement)
        self.timer_camera_2.timeout.connect(
            self.show_displacement)  # 若定时器结束，则调用show_camera()

    '''槽函数之一'''

    def button_open_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(
                self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(
                    self,
                    'warning',
                    "请检查相机于电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(50)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_video.clear()  # 清空视频显示区域
            self.button_open.setText('打开相机')

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        self.image = cv2.flip(self.image, 1)
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(
            show.data, show.shape[1], show.shape[0],
            QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_video.setPixmap(
            QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def hsv2rgb(h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    def button_get_displacement(self):
        if self.timer_camera_2.isActive() == False:  # 若定时器未启动
            # 水平噪声、数值噪声
            self.stream = VideoGear(
                source=0,
                stabilize=self.para['is_anti_shake'],
                **self.solution['options']).start(
                )  # To open any valid video stream(for e.g device at 0 index)

            self.timer_camera_2.start(50)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
            self.button_getdisplacement.setText('关闭displacement')
        else:
            self.timer_camera_2.stop()  # 关闭定时器
            self.stream.stop()
            self.label_video_2.clear()  # 清空视频显示区域
            self.button_getdisplacement.setText('打开displacement')

    def show_displacement(self):
        frame_pre = self.stream.read()
        print("video resolution is (height, width, channel) : ",frame_pre.shape)
        bgr_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)
        # initial hsv
        hsv = np.zeros_like(
            frame_pre[
            self.solution['roi_rect'][2]+self.para['Noffset']:
            self.solution['roi_rect'][3]-self.para['Noffset'],
            self.solution['roi_rect'][0]+self.para['Noffset']:
            self.solution['roi_rect'][1]-self.para['Noffset']])  
        hsv[...,1] = 255 #saturation is full
        blank = np.zeros_like(frame_pre)
        frame_cur = self.stream.read()
        bgr_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY) # change in to gray
    
        # 测量两幅同样图像
        if self.para['is_ref_refresh']:
            bgr_pre = bgr_cur
        # 图像裁切
        roi_bgr_pre = bgr_pre[self.solution['roi_rect'][2]:self.solution['roi_rect'][3],self.solution['roi_rect'][0]:self.solution['roi_rect'][1]]
        roi_bgr_cur = bgr_cur[self.solution['roi_rect'][2]:self.solution['roi_rect'][3],self.solution['roi_rect'][0]:self.solution['roi_rect'][1]]

        # calculate flow
        
        flow_frame = cv2.calcOpticalFlowFarneback(roi_bgr_pre,roi_bgr_cur, None,
                                        self.para['pyr_scale'],
                                        self.para['levels'],
                                        self.para['winsize'],
                                        self.para['iterations'],
                                        self.para['poly_n'],
                                        self.para['poly_sigma'],
                                        self.flags)
      
       
        # 处理mag ang函数
        mag, ang = cv2.cartToPolar(flow_frame[...,0], flow_frame[...,1])  # orginal flow
        mag = mag[self.para['Noffset']:mag.shape[0]-self.para['Noffset'], self.para['Noffset']:(mag.shape[1]-self.para['Noffset'])]
        ang = ang[self.para['Noffset']:ang.shape[0]-self.para['Noffset'], self.para['Noffset']:(ang.shape[1]-self.para['Noffset'])]
        # mag_mean = cv.mean(mag)[0]
        mag_mean = 0
        mag_sft = abs(mag - mag_mean)  # shifted magnitude to elimiate noise
        
        mag_enhanced = zeros_like(mag_sft)
        cv2.min(mag_sft,self.para['mag_ceiling'],mag_enhanced)  # enhance flow, ceiling and flooring
        cv2.max(mag_enhanced,self.para['mag_floor'],mag_enhanced)
        hsv[...,0] = (ang + self.para['hue'])*180/np.pi/2 # color space related to angle 
        hsv[...,2] = cv2.normalize(mag_enhanced,None,0,255,cv2.NORM_MINMAX)
        # hsv[...,2] = cv.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr_flow_enhanced = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        # 程序结果的问题出在这几行
        # 对结果矩阵偏移的修正
        horMat = np.multiply(mag, np.cos(ang))
        verMat = np.multiply(mag, np.sin(ang))

        
        # image emerge with enhanced flow
        flow_blend_enhance = cv2.addWeighted(frame_cur[self.solution['roi_rect'][2]+self.para['Noffset']:self.solution['roi_rect'][3]-self.para['Noffset'],self.solution['roi_rect'][0]+self.para['Noffset']:self.solution['roi_rect'][1]-self.para['Noffset']], 1-self.para['alpha'] ,bgr_flow_enhanced,  self.para['alpha'], 0)
        frame_blend = cv2.addWeighted(frame_cur, 1-self.para['alpha'] ,blank,  self.para['alpha'], 0)#frame_cur.copy()
        frame_blend[self.solution['roi_rect'][2]+self.para['Noffset']:self.solution['roi_rect'][3]-self.para['Noffset'],self.solution['roi_rect'][0]+self.para['Noffset']:self.solution['roi_rect'][1]-self.para['Noffset']] = flow_blend_enhance
        
        #palette
        palette=np.zeros((512,512,3),np.uint8)
        mag_norm = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        mag_min,mag_max,min_indx,max_indx=cv2.minMaxLoc(mag)
        ang_mean = cv2.mean(ang*mag/mag_mean)[0]

        _,hor_MaxNoise, _, _ = cv2.minMaxLoc(np.multiply(mag, np.cos(ang))-0*np.ones(horMat.shape))
        _,ver_MaxNoise, _, _ = cv2.minMaxLoc(np.multiply(mag, np.sin(ang)))

        # 计算水平噪声和竖直噪声
        hor_Noise = np.average(np.multiply(mag, np.cos(ang))-0*np.ones(horMat.shape))
        ver_Noise = np.average(np.multiply(mag, np.sin(ang)))
        # # 绝对值计算水平噪声和竖直噪声
        # hor_AbsNoise = np.average(np.abs(horMat))
        # ver_AbsNoise = np.average(np.abs(verMat))


        cv2.putText(palette, "max="+str(mag.max()), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_mean="+str("%.7f"%mag_mean), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_max="+str(mag_max), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_min="+str(mag_min), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "hor_Noise="+str(hor_Noise), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "ver_Noise="+str(ver_Noise), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "hor_MaxNoise="+str(hor_MaxNoise), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "ver_MaxNoise="+str(ver_MaxNoise), (0, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
        # result flow image
        cv2.imshow('Orignal',cv2.resize(frame_cur, (self.para['result_RES'][0], self.para['result_RES'][1])))
        cv2.imshow("Palette", palette)
        cv2.imshow('Area of Intrest Blended', cv2.resize(frame_blend  , (self.para['result_RES'][0], self.para['result_RES'][1])))
        
        self.count += 1
        # key = cv2.waitKey()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = MyMainWindow()  # 实例化MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过