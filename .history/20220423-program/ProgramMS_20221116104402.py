import numpy as np 
import cv2 as cv
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

#hsv transform to rgb format
def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

f = open('20220423-program/bosparameters.json', 'r')
content = f.read()
para = json.loads(content)
flags = cv.OPTFLOW_FARNEBACK_GAUSSIAN
# 方案0大分辨率，方案1小分辨率
solution = para['solution'][1]

# 水平噪声、数值噪声
stream = VideoGear(source=1, stabilize=para['is_anti_shake'], **solution['options']).start() # To open any valid video stream(for e.g device at 0 index)

# pre-read frame
frame_pre = stream.read()
print("video resolution is (height, width, channel) : ",frame_pre.shape)
bgr_pre = cv.cvtColor(frame_pre, cv.COLOR_BGR2GRAY)
# initial hsv
hsv = np.zeros_like(
    frame_pre[
    solution['roi_rect'][2]+para['Noffset']:
    solution['roi_rect'][3]-para['Noffset'],
    solution['roi_rect'][0]+para['Noffset']:
    solution['roi_rect'][1]-para['Noffset']])  
hsv[...,1] = 255 #saturation is full
blank = np.zeros_like(frame_pre)

# video loop
count = 0
print("begin stream!")
print("video resolution is (height, width, channel) : ",frame_pre.shape)
# 程序执行区域
timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("tVariance_" + timeNow)
with open("tVariance_" + timeNow + '/Variance.csv','w',newline='')as csv_file:
    while(True):
    # read current frame from cap
        frame_cur = stream.read()
        bgr_cur = cv.cvtColor(frame_cur, cv.COLOR_BGR2GRAY) # change in to gray
    
        # 测量两幅同样图像
        if para['is_ref_refresh']:
            bgr_pre = bgr_cur
        # 图像裁切
        roi_bgr_pre = bgr_pre[solution['roi_rect'][2]:solution['roi_rect'][3],solution['roi_rect'][0]:solution['roi_rect'][1]]
        roi_bgr_cur = bgr_cur[solution['roi_rect'][2]:solution['roi_rect'][3],solution['roi_rect'][0]:solution['roi_rect'][1]]

        # calculate flow
        
        flow_frame = cv.calcOpticalFlowFarneback(roi_bgr_pre,roi_bgr_cur, None,
                                        para['pyr_scale'],
                                        para['levels'],
                                        para['winsize'],
                                        para['iterations'],
                                        para['poly_n'],
                                        para['poly_sigma'],
                                        flags)
        if count == 0:
            flow = np.zeros_like(flow_frame)
        # mag_frame 和 ang_frame 都是每幅图像的矢量，加到
        
        flow = (flow + flow_frame)/2
        # flow = (flow*count + flow_frame)/(count+1)
        # flow = flow_frame
        
        # 处理mag ang函数
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])  # orginal flow
        mag = mag[para['Noffset']:mag.shape[0]-para['Noffset'], para['Noffset']:(mag.shape[1]-para['Noffset'])]
        ang = ang[para['Noffset']:ang.shape[0]-para['Noffset'], para['Noffset']:(ang.shape[1]-para['Noffset'])]
        # mag_mean = cv.mean(mag)[0]
        mag_mean = 0
        mag_sft = abs(mag - mag_mean)  # shifted magnitude to elimiate noise
        
        mag_enhanced = zeros_like(mag_sft)
        cv.min(mag_sft,para['mag_ceiling'],mag_enhanced)  # enhance flow, ceiling and flooring
        cv.max(mag_enhanced,para['mag_floor'],mag_enhanced)
        hsv[...,0] = (ang + para['hue'])*180/np.pi/2 # color space related to angle 
        hsv[...,2] = cv.normalize(mag_enhanced,None,0,255,cv.NORM_MINMAX)
        # hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr_flow_enhanced = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        
        # 程序结果的问题出在这几行
        # 对结果矩阵偏移的修正
        horMat = np.multiply(mag, np.cos(ang))
        verMat = np.multiply(mag, np.sin(ang))

        
        # image emerge with enhanced flow
        flow_blend_enhance = cv.addWeighted(frame_cur[solution['roi_rect'][2]+para['Noffset']:solution['roi_rect'][3]-para['Noffset'],solution['roi_rect'][0]+para['Noffset']:solution['roi_rect'][1]-para['Noffset']], 1-para['alpha'] ,bgr_flow_enhanced,  para['alpha'], 0)
        frame_blend = cv.addWeighted(frame_cur, 1-para['alpha'] ,blank,  para['alpha'], 0)#frame_cur.copy()
        frame_blend[solution['roi_rect'][2]+para['Noffset']:solution['roi_rect'][3]-para['Noffset'],solution['roi_rect'][0]+para['Noffset']:solution['roi_rect'][1]-para['Noffset']] = flow_blend_enhance
        
        #palette
        palette=np.zeros((512,512,3),np.uint8)
        mag_norm = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        mag_min,mag_max,min_indx,max_indx=cv.minMaxLoc(mag)
        ang_mean = cv.mean(ang*mag/mag_mean)[0]

        # # 绘制矢量箭头方案一：opencv-arrowedline
        # col, row = np.meshgrid(np.arange(mag.shape[0]),np.arange(mag.shape[1]))
        # coord = np.stack((col,row), axis=2)
        # for i in np.arange(mag.size):
        #     x = i % mag.shape[0]
        #     y = int(i / mag.shape[1])
        #     # 这个函数第二个元组不能有小数点
        #     cv.arrowedLine(frame_blend, (x, y),(x+verMat[x,y], y+horMat[x,y]), (255, 0, 0), 2, 9, 0, 0.3)  # 画箭头

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

        _,hor_MaxNoise, _, _ = cv.minMaxLoc(np.multiply(mag, np.cos(ang))-0*np.ones(horMat.shape))
        _,ver_MaxNoise, _, _ = cv.minMaxLoc(np.multiply(mag, np.sin(ang)))

        # 计算水平噪声和竖直噪声
        hor_Noise = np.average(np.multiply(mag, np.cos(ang))-0*np.ones(horMat.shape))
        ver_Noise = np.average(np.multiply(mag, np.sin(ang)))
        # # 绝对值计算水平噪声和竖直噪声
        # hor_AbsNoise = np.average(np.abs(horMat))
        # ver_AbsNoise = np.average(np.abs(verMat))


        cv.putText(palette, "max="+str(mag.max()), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "mag_mean="+str("%.7f"%mag_mean), (0, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "mag_max="+str(mag_max), (0, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "mag_min="+str(mag_min), (0, 120), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "hor_Noise="+str(hor_Noise), (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "ver_Noise="+str(ver_Noise), (0, 180), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "hor_MaxNoise="+str(hor_MaxNoise), (0, 210), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.putText(palette, "ver_MaxNoise="+str(ver_MaxNoise), (0, 240), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        timeNow1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # max min min variance 
        # 获取一个csv对象进行内容写入
        writer=csv.writer(csv_file)
        # writerow 写入一行数据
        writeDatum = [timeNow1, mag_mean, mag_max, mag_min, hor_Noise, ver_Noise, hor_MaxNoise, ver_MaxNoise]
        writer.writerow(writeDatum)

        count = count + 1
    
        # result flow image
        cv.imshow('Orignal',cv.resize(frame_cur, (para['result_RES'][0], para['result_RES'][1])))
        cv.imshow("Palette", palette)
        # cv.imshow('Blender', cv.resize(bgr_blend , (para['result_RES'][0], para['result_RES'][1])))     
        cv.imshow('Area of Intrest Blended', cv.resize(frame_blend  , (para['result_RES'][0], para['result_RES'][1])))
        
        key = cv.waitKey(10)
        if key==114: # if input key 'r', refresh compare image
            #打印时间戳保存
            # timeNow1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # os.makedirs("Refresh_" + timeNow1)
            bgr_pre = bgr_cur
            # cv.imwrite(timeNow +'/Orignal.jpg',frame_cur)
            # cv.imwrite(timeNow +'/Area_of_Intrest_Blended.jpg', frame_blend)
            # cv.imwrite(timeNow1 +"/Palette.jpg", palette)
            # cv.imwrite(timeNow1 +'/Blender.jpg', bgr_blend)
            # print("the image is refreshed")

        # if input key 's', refresh compare image
        if key==115:
            # 绘制magnitude图片
            print("mag.shape: ", np.shape(mag))
            timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("save_" + timeNow)
            np.savetxt("save_" + timeNow +"/mag.txt", mag)
            np.savetxt("save_" + timeNow +"/angle.txt", ang)