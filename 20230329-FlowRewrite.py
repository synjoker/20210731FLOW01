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

#hsv transform to rgb format
def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

# 自定义卷积函数
def my_conv(input, kernel, step):
    output_size_0 = int((len(input) - len(kernel)) / step + 1)   # 输出结果的第0维长度
    output_size_1 = int((len(input[0]) - len(kernel[0])) / step + 1)   # 输出结果的第1维长度
    res = np.zeros([output_size_0, output_size_1], np.float32)

    for i in range(len(res)):
        for j in range(len(res[0])):
            a = input[i*step:i*step + len(kernel), j*step: j*step + len(kernel)]  # 从输入矩阵中取出子矩阵
            b = a * kernel  # 对应元素相乘
            res[i][j] = b.sum()   
    return res

# 自定义鼠标双击事件——获取图像坐标
# 1. 对图像进行操作，同时生成一副新的图像；
# maybe需要多线程
# 2. 对视频实施爬取，读取连续图像，并且可以生成曲线图；
# 还是需要多线程

frame_cur = np.zeros((512,512,3), np.uint8)
def get_coord(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(frame_cur,(x,y),100,(255,0,0),-1)
        if cv2.waitKey() & 0xFF ==27 :
            print("Continue Play!")
        
# Create a black image, a window and bind the function to window
cv2.namedWindow('Orignal')
cv2.setMouseCallback('Orignal',get_coord)

#control parameters
is_save_stream = False
is_anti_shake = False
is_ref_refresh = is_anti_shake
is_change_size = True

# default parameters  
# calculation parameters
winsize = 64
mag_ceiling = 0.5
mag_floor = mag_ceiling/5

# camer & image parameters
alpha = 0.8 
hue = 90                     
RES=(1920,1080)                 # camera resolution
# roi_rect = [0,RES[0],0,RES[1]]  # region of interesting
roi_rect = [200,600,115,420] # 转置前

result_RES = (1080, 720)         # flow image resolution
# options = {"CAP_PROP_FRAME_WIDTH":2592, "CAP_PROP_FRAME_HEIGHT":1944, "CAP_PROP_FPS":30}
options = {"CAP_PROP_FRAME_WIDTH":800, "CAP_PROP_FRAME_HEIGHT":600, "CAP_PROP_FPS":30}
# stable calculation parameters
pyr_scale = 0.5
levels = 2
iterations = 1
poly_n = 5
poly_sigma = 1.1
flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
Noffset = 30 # 消除光流算法偏差的位移量
# drawing buffer
ax = []
ay1 = []
ay2 = []
#plt.ion()

# 水平噪声、数值噪声
# stream = VideoGear(source="./WIN_20230329_16_13_08_Pro.mp4", stabilize= is_anti_shake , resolution=RES, **options).start()
stream = VideoGear(source="./WIN_20230329_16_13_08_Pro.mp4", stabilize= is_anti_shake , resolution=RES, **options).start()

# pre-read frame
frame_pre = stream.read()
print("video resolution is (height, width, channel) : ",frame_pre.shape)
bgr_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)
# initial hsv
hsv = np.zeros_like(frame_pre)  
# hsv = np.zeros_like(frame_pre[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]])  
hsv[...,1] = 255 #saturation is full
blank = np.zeros_like(frame_pre)

# video loop
count = 0
print("begin stream!")

while True:
    try:
        frame_cur = stream.read()
        if frame_cur is None:
            print("video has played over!")
            break
        # 跳过不需要的帧
        count += 1
        if count < 60:
            print("%d times"% count)
            continue
        bgr_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY) # change in to gray
        # 测量两幅同样图像
        if is_ref_refresh:
            bgr_pre = bgr_cur
        
        if is_change_size:
            # 改变图像大小
            # resize_resolution = [1296, 972]    
            resize_resolution = [800, 600]
            frame_pre   = cv2.resize(frame_pre ,resize_resolution)
            frame_cur   = cv2.resize(frame_cur ,resize_resolution)
            bgr_pre     = cv2.resize(bgr_pre ,resize_resolution)
            bgr_cur     = cv2.resize(bgr_cur ,resize_resolution)
            hsv         = cv2.resize(hsv ,resize_resolution)
            blank       = cv2.resize(blank ,resize_resolution)
        
        # 图像裁切
        roi_bgr_pre = bgr_pre
        roi_bgr_cur = bgr_cur
        
        flow = cv2.calcOpticalFlowFarneback(roi_bgr_pre,roi_bgr_cur, None,
                                        pyr_scale,
                                        levels,
                                        winsize,
                                        iterations,
                                        poly_n,
                                        poly_sigma,
                                        flags)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])  # orginal flow
        
        # 绘制矢量箭头方案一：opencv-arrowedline
        min_arrowline_threshold = 1.5   # 矢量箭头阈值下限（与矢量平均值相除）
        max_arrowline_threshold = 10    # 矢量箭头阈值下限
        step = 20                       # 卷积模板步长
        half_step = int(step/2)         # 卷积模板的半步长
        # 1. 确定要画的mag及其坐标
        mag_mean = cv2.mean(mag)[0]
        mag = np.where(mag > min_arrowline_threshold * mag_mean, mag, 0)
        mag = np.where(mag < max_arrowline_threshold * mag_mean, mag, 0)
        kernel = np.ones((half_step, half_step)) / half_step**2
        mag_conv = my_conv(mag, kernel, step)
        ang_conv = my_conv(ang, kernel, step)
        
        horMat = np.multiply(mag_conv, np.cos(ang_conv))
        verMat = np.multiply(mag_conv, np.sin(ang_conv))
        # 2. 卷积形成矢量箭头
        for index, value in np.ndenumerate(mag_conv):
            if value != 0.0:
                # print(index, value, horMat[index], verMat[index])
                cv2.arrowedLine(frame_cur, (index[1]*step, index[0]*step) ,(index[1]*step+int(horMat[index]*step), index[0]*step+int(verMat[index]*step)), (255, 0, 0), 2, 9, 0, 0.3)  # 画箭头
        
        
        mag_sft = abs(mag - mag_mean)  # shifted magnitude to elimiate noise
        hsv[...,0] = (ang + hue)*180/np.pi/2 # color space related to angle 
        hsv[...,2] = cv2.normalize(mag_sft,None,0,255,cv2.NORM_MINMAX) 
        bgr_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        #enhanced flow
        mag_enhanced = zeros_like(mag_sft)
        cv2.min(mag_sft,mag_ceiling,mag_enhanced)  # enhance flow, ceiling and flooring
        cv2.max(mag_enhanced,mag_floor,mag_enhanced)
        hsv[...,0] = (ang + hue)*180/np.pi/2 # color space related to angle 
        hsv[...,2] = cv2.normalize(mag_enhanced,None,0,255,cv2.NORM_MINMAX)
        bgr_flow_enhanced = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # image emerge with enhanced flow
        flow_blend_enhance = cv2.addWeighted(frame_cur, 1-alpha ,bgr_flow_enhanced,  alpha, 0)
        bgr_blend = cv2.addWeighted(frame_cur, 1-alpha ,bgr_flow,  alpha, 0)
        frame_blend = flow_blend_enhance
        
        if is_ref_refresh:
            bgr_pre = bgr_cur
        
        #palette
        palette=np.zeros((512,512,3),np.uint8)
        mag_norm = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        mag_min,mag_max,min_indx,max_indx=cv2.minMaxLoc(mag)
        ang_mean = cv2.mean(ang*mag/mag_mean)[0]

        # 对结果矩阵偏移的修正
        horMat = np.multiply(mag, np.cos(ang))
        verMat = np.multiply(mag, np.sin(ang))

        # 计算水平噪声最大值和竖直噪声最大值
        _,hor_MaxNoise, _, _ = cv2.minMaxLoc(np.multiply(mag, np.cos(ang)))
        _,ver_MaxNoise, _, _ = cv2.minMaxLoc(np.multiply(mag, np.sin(ang)))


        # 计算水平噪声和竖直噪声
        hor_Noise = np.average(np.multiply(mag, np.cos(ang)))
        ver_Noise = np.average(np.multiply(mag, np.sin(ang)))
        # # 绝对值计算水平噪声和竖直噪声
        # hor_AbsNoise = np.average(np.abs(horMat))
        # ver_AbsNoise = np.average(np.abs(verMat))

 
    except Exception:
        # read current frame from cap
        print("something wrong!")
        stream.stop()  
        raise
    
    else:
        
        print("nothing wrong!")
        print("%d times"% count)
        cv2.putText(palette, "max="+str(mag.max()), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_mean="+str("%.7f"%mag_mean), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_max="+str(mag_max), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "mag_min="+str(mag_min), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "hor_Noise="+str(hor_Noise), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "ver_Noise="+str(ver_Noise), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "hor_MaxNoise="+str(hor_MaxNoise), (0, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(palette, "ver_MaxNoise="+str(ver_MaxNoise), (0, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # # result flow image
        cv2.imshow('Orignal',cv2.resize(frame_cur, (result_RES[0], result_RES[1])))
        cv2.imshow("Palette", palette)
        cv2.imshow('Blender', cv2.resize(bgr_blend , (result_RES[0], result_RES[1])))
        cv2.imshow('Area of Intrest Blended', cv2.resize(frame_blend  , (result_RES[0], result_RES[1])))
        
        key = cv2.waitKey(1)
        
        if key == ord("r"):
            bgr_pre = bgr_cur
            print("Background Refresh!")
            
        if key == ord("q"):
            print("KeyboardInterrupt!")
            break

  
cv2.destroyAllWindows()
