from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
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
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera_2 = QtCore.QTimer()
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.slot_init()  # 初始化槽函数

        #hsv transform to rgb format
        f = open('20220423-program/bosparameters.json', 'r')
        content = f.read()
        self.para = json.loads(content)
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        # 方案0大分辨率，方案1小分辨率
        self.solution = self.para['solution'][1]
 
    '''初始化所有槽函数'''
    def slot_init(self):
        self.button_open.clicked.connect(self.button_open_clicked)  # 若该按键被点击，则调用button_open_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_getdisplacement.clicked.connect(self.button_get_displacement)
        self.timer_camera_2.timeout.connect(self.show_displacement)  # 若定时器结束，则调用show_camera()
    
    '''槽函数之一'''
 
    def button_open_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
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
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_video.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
    
    def hsv2rgb(h,s,v):
            return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

    def button_get_displacement(self):
        if self.timer_camera_2.isActive() == False:  # 若定时器未启动
            # 水平噪声、数值噪声
            self.stream = VideoGear(source=0, stabilize=self.para['is_anti_shake'], **self.solution['options']).start() # To open any valid video stream(for e.g device at 0 index)          

            self.timer_camera_2.start(50)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
            self.button_getdisplacement.setText('关闭displacement')
        else:
            self.timer_camera_2.stop()  # 关闭定时器
            self.stream.stop()
            self.label_video_2.clear()  # 清空视频显示区域
            self.button_getdisplacement.setText('打开displacement')

    def show_displacement(self):
        frame_pre = self.stream.read()
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





        show = cv2.resize(frame_pre, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_video_2.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImag

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = MyMainWindow()  # 实例化MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过