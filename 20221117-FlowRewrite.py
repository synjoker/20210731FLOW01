import numpy as np
import cv2
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


class BOSdetect:

    def __init__(self):
        self.key = 0
        self.count = 0
        pass

    #hsv transform to rgb format
    def hsv2rgb(h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    def LoadJson(self, jsonpath='20220423-program/bosparameters.json'):
        f = open(jsonpath, 'r')
        content = f.read()
        self.para = json.loads(content)
        self.flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        # 方案0大分辨率，方案1小分辨率
        self.solution = self.para['solution'][1]
        # print(self.para.keys())

    def PicPreProcess(self, cameraNum=0):
        # 水平噪声、数值噪声
        self.stream = VideoGear(
            source=cameraNum,
            stabilize=self.para['is_anti_shake'],
            **self.solution['options']).start(
            )  # To open any valid video stream(for e.g device at 0 index)
        # pre-read frame
        self.frame_pre = self.stream.read()
        print("video reself.solution is (height, width, channel) : ",
              self.frame_pre.shape)
        self.bgr_pre = cv2.cvtColor(self.frame_pre, cv2.COLOR_BGR2GRAY)
        # initial hsv
        self.hsv = np.zeros_like(
            self.frame_pre[self.solution['roi_rect'][2] +
                           self.para['Noffset']:self.solution['roi_rect'][3] -
                           self.para['Noffset'], self.solution['roi_rect'][0] +
                           self.para['Noffset']:self.solution['roi_rect'][1] -
                           self.para['Noffset']])
        self.hsv[..., 1] = 255  #saturation is full
        self.blank = np.zeros_like(self.frame_pre)

    def GreypicView(self):
        frame_cur = self.stream.read()
        self.bgr_cur = cv2.cvtColor(frame_cur,
                                    cv2.COLOR_BGR2GRAY)  # change in to gray

        # 测量两幅同样图像
        if self.para['is_ref_refresh']:
            self.bgr_pre = self.bgr_cur
        # 图像裁切
        roi_bgr_pre = self.bgr_pre[
            self.solution['roi_rect'][2]:self.solution['roi_rect'][3],
            self.solution['roi_rect'][0]:self.solution['roi_rect'][1]]
        roi_bgr_cur = self.bgr_cur[
            self.solution['roi_rect'][2]:self.solution['roi_rect'][3],
            self.solution['roi_rect'][0]:self.solution['roi_rect'][1]]

        # calculate self.flow
        flow_frame = cv2.calcOpticalFlowFarneback(
            roi_bgr_pre, roi_bgr_cur, None, self.para['pyr_scale'],
            self.para['levels'], self.para['winsize'], self.para['iterations'],
            self.para['poly_n'], self.para['poly_sigma'], self.flags)
        if self.count == 0:
            self.flow = np.zeros_like(flow_frame)
        # mag_frame 和 ang_frame 都是每幅图像的矢量，加到
        self.flow = (self.flow + flow_frame) / 2
        # self.flow = (self.flow*count + flow_frame)/(count+1)
        # self.flow = flow_frame

        # 处理mag ang函数
        self.mag, self.ang = cv2.cartToPolar(self.flow[..., 0],
                                             self.flow[...,
                                                       1])  # orginal self.flow
        self.mag = self.mag[self.para['Noffset']:self.mag.shape[0] -
                            self.para['Noffset'],
                            self.para['Noffset']:(self.mag.shape[1] -
                                                  self.para['Noffset'])]
        self.ang = self.ang[self.para['Noffset']:self.ang.shape[0] -
                            self.para['Noffset'],
                            self.para['Noffset']:(self.ang.shape[1] -
                                                  self.para['Noffset'])]
        # mag_mean = cv2.mean(mag)[0]
        self.mag_mean = 0
        mag_sft = abs(self.mag -
                      self.mag_mean)  # shifted magnitude to elimiate noise

        mag_enhanced = zeros_like(mag_sft)
        cv2.min(mag_sft, self.para['mag_ceiling'],
                mag_enhanced)  # enhance self.flow, ceiling and flooring
        cv2.max(mag_enhanced, self.para['mag_floor'], mag_enhanced)
        self.hsv[..., 0] = (self.ang + self.para['hue']
                            ) * 180 / np.pi / 2  # color space related to angle
        self.hsv[..., 2] = cv2.normalize(mag_enhanced, None, 0, 255,
                                         cv2.NORM_MINMAX)
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        self.bgr_flow_enhanced = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        # 程序结果的问题出在这几行
        # 对结果矩阵偏移的修正
        self.horMat = np.multiply(self.mag, np.cos(self.ang))
        self.verMat = np.multiply(self.mag, np.sin(self.ang))

        # image emerge with enhanced self.flow
        self.flow_blend_enhance = cv2.addWeighted(
            frame_cur[self.solution['roi_rect'][2] +
                      self.para['Noffset']:self.solution['roi_rect'][3] -
                      self.para['Noffset'], self.solution['roi_rect'][0] +
                      self.para['Noffset']:self.solution['roi_rect'][1] -
                      self.para['Noffset']], 0.6,
            self.bgr_flow_enhanced, self.para['alpha'], 0)
        self.frame_blend = cv2.addWeighted(frame_cur, 0.6,
                                           self.blank, 0.1,
                                           0)  #frame_cur.copy()
        self.frame_blend[self.solution['roi_rect'][2] +
                         self.para['Noffset']:self.solution['roi_rect'][3] -
                         self.para['Noffset'], self.solution['roi_rect'][0] +
                         self.para['Noffset']:self.solution['roi_rect'][1] -
                         self.para['Noffset']] = self.flow_blend_enhance

        # result self.flow image
        cv2.imshow(
            'Orignal',
            cv2.resize(
                frame_cur,
                (self.para['result_RES'][0], self.para['result_RES'][1])))

    def PattleView(self):
        #palette
        palette = np.zeros((512, 512, 3), np.uint8)
        mag_norm = cv2.normalize(self.mag, None, 0, 255, cv2.NORM_MINMAX)
        self.mag_min, self.mag_max, self.min_indx, self.max_indx = cv2.minMaxLoc(
            self.mag)
        ang_mean = cv2.mean(self.ang * self.mag / self.mag_mean)[0]

        _, self.hor_MaxNoise, _, _ = cv2.minMaxLoc(
            np.multiply(self.mag, np.cos(self.ang)) -
            0 * np.ones(self.horMat.shape))
        _, self.ver_MaxNoise, _, _ = cv2.minMaxLoc(
            np.multiply(self.mag, np.sin(self.ang)))

        # 计算水平噪声和竖直噪声
        self.hor_Noise = np.average(
            np.multiply(self.mag, np.cos(self.ang)) -
            0 * np.ones(self.horMat.shape))
        self.ver_Noise = np.average(np.multiply(self.mag, np.sin(self.ang)))
        # # 绝对值计算水平噪声和竖直噪声
        # hor_AbsNoise = np.average(np.abs(horMat))
        # ver_AbsNoise = np.average(np.abs(verMat))

        cv2.putText(palette, "max=" + str(self.mag.max()), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_mean=" + str("%.7f" % self.mag_mean),
                    (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_max=" + str(self.mag_max), (0, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_min=" + str(self.mag_min), (0, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "hor_Noise=" + str(self.hor_Noise), (0, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "ver_Noise=" + str(self.ver_Noise), (0, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "hor_MaxNoise=" + str(self.hor_MaxNoise),
                    (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),
                    2)
        cv2.putText(palette, "ver_MaxNoise=" + str(self.ver_MaxNoise),
                    (0, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),
                    2)
        cv2.imshow("Palette", palette)

    def BlendView(self):
        # cv2.imshow('Blender', cv2.resize(bgr_blend , (self.para['result_RES'][0], self.para['result_RES'][1])))
        cv2.imshow(
            'Area of Intrest Blended',
            cv2.resize(
                self.frame_blend,
                (self.para['result_RES'][0], self.para['result_RES'][1])))

    def QuiverView(self):
        # # 绘制矢量箭头方案一：opencv-arrowedline
        # col, row = np.meshgrid(np.arange(mag.shape[0]),np.arange(mag.shape[1]))
        # coord = np.stack((col,row), axis=2)
        # for i in np.arange(mag.size):
        #     x = i % mag.shape[0]
        #     y = int(i / mag.shape[1])
        #     # 这个函数第二个元组不能有小数点
        #     cv2.arrowedLine(frame_blend, (x, y),(x+verMat[x,y], y+horMat[x,y]), (255, 0, 0), 2, 9, 0, 0.3)  # 画箭头

        # # 方案二：quiver-matplotlib
        # col, row = np.meshgrid(np.arange(mag.shape[0]),np.arange(mag.shape[1]))
        # verMat = np.transpose(verMat)
        # horMat = np.transpose(horMat)
        # plt.ion()  #打开交互模式
        # ax = plt.axes()
        # inter = 50
        # x = sum(col[0:col.shape[0]-1:inter, 0:col.shape[1]-1:inter].tolist(),[])
        # y = sum(row[0:col.shape[0]-1:inter, 0:col.shape[1]-1:inter].tolist(),[])
        # x_dis = sum(verMat[0:col.shape[0]-1:inter, 0:col.shape[1]-1:inter].tolist(),[])
        # x_dis = np.multiply(x_dis, -1)
        # x_dis = np.flip(x_dis, axis=0)  #行反转
        # y_dis = sum(horMat[0:col.shape[0]-1:inter, 0:col.shape[1]-1:inter].tolist(),[])
        # ax.quiver(y,x,y_dis,x_dis,color=(1, 0, 0, 0.3),angles='xy', scale_units='xy', scale=0.002)
        # ax.grid()
        # ax.set_xlabel('X')
        # ax.set_xlim(0, mag.shape[1]-1)
        # ax.set_ylabel('Y')
        # ax.set_ylim(0, mag.shape[1]-1)
        # plt.show()
        # plt.pause(0.6)
        # plt.clf()  #清除图像
        # # 计算水平噪声最大值和竖直噪声最大值
        # # 上面作图转换了 下面还原回来
        # verMat = np.transpose(verMat)
        # horMat = np.transpose(horMat)
        pass

    def CsvRecord(self):
        timeNow1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # max min min variance
        # 获取一个csv对象进行内容写入
        writer = csv.writer(self.csv_file)
        # writerow 写入一行数据
        writeDatum = [
            timeNow1, self.mag_mean, self.mag_max, self.mag_min,
            self.hor_Noise, self.ver_Noise, self.hor_MaxNoise,
            self.ver_MaxNoise
        ]
        writer.writerow(writeDatum)

        self.count = self.count + 1

    def CameraLoop(self):
        pass

    def KeyboardRecord(self):
        if self.key == ord('r'):  # if input key 'r', refresh compare image
            #打印时间戳保存
            # timeNow1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # os.makedirs("Refresh_" + timeNow1)
            self.bgr_pre = self.bgr_cur
            # cv2.imwrite(timeNow +'/Orignal.jpg',frame_cur)
            # cv2.imwrite(timeNow +'/Area_of_Intrest_Blended.jpg', frame_blend)
            # cv2.imwrite(timeNow1 +"/Palette.jpg", palette)
            # cv2.imwrite(timeNow1 +'/Blender.jpg', bgr_blend)
            # print("the image is refreshed")

        # if input key 's', refresh compare image
        if self.key == ord('s'):
            # 绘制magnitude图片
            print("mag.shape: ", np.shape(self.mag))
            timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("save_" + timeNow)
            np.savetxt("save_" + timeNow + "/mag.txt", self.mag)
            np.savetxt("save_" + timeNow + "/angle.txt", self.ang)

    def OpencvShow(self):
        pass

    def PartialDifferentialEquation(self):
        pass


if __name__ == "__main__":
    bos = BOSdetect()
    bos.count = 0
    jsonpath = '20220423-program/bosparameters.json'
    bos.LoadJson()
    bos.PicPreProcess()
    print("begin stream!")
    print("video reself.solution is (height, width, channel) : ",
          bos.frame_pre.shape)
    # 程序执行区域
    timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("tVariance_" + timeNow)
    with open("tVariance_" + timeNow + '/Variance.csv', 'w',
              newline='') as bos.csv_file:
        while (True):
            # read current frame from cap
            bos.GreypicView()
            bos.PattleView()
            bos.BlendView()
            bos.CsvRecord()
            bos.KeyboardRecord()
            bos.key = cv2.waitKey(10)
