#Flow
# problems: 1. video lag 2. stream end 3. anti-shake 4. simulation of the method 
# refresh parameters 1. change in 'parameters.ini' 2. press 'r'
# to do log information, max position and angle, average angle
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
from mpl_toolkits.mplot3d import Axes3D

#hsv transform to rgb format
def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

#control parameters
is_save_stream = False
is_ref_refresh = False
is_anti_shake = False

# default parameters  
# calculation parameters
winsize = 64
mag_ceiling = 0.5
mag_floor = mag_ceiling/5

# camer & image parameters
alpha = 0.8 
hue = 90                     
RES=(1920,1080)                 # camera resolution
roi_rect = [0+100,RES[0]-100,0,RES[1]]  # region of interesting
# roi_rect = [int(RES[0]*0.4), int(RES[0]*0.6), int(RES[1]*0.4), int(RES[1]*0.6)]
result_RES = (1920, 1080)         # flow image resolution

# stable calculation parameters
pyr_scale = 0.5
levels = 2
iterations = 1
poly_n = 5
poly_sigma = 1.1
flags = cv.OPTFLOW_FARNEBACK_GAUSSIAN

# drawing buffer
ax = []
ay1 = []
ay2 = []

# formatting parameters as dictionary attributes
options = {"CAP_PROP_FRAME_WIDTH":2592, "CAP_PROP_FRAME_HEIGHT":1944, "CAP_PROP_FPS":30}


# 水平噪声、数值噪声
# stream = VideoGear(source=1, resolution=RES, stabilize = True).start()
stream = VideoGear(source=1, resolution=RES, **options).start()
# stream = VideoGear(source=0, resolution=(2592, 1944)).start()

# pre-read frame
frame_pre = stream.read()
bgr_pre = cv.cvtColor(frame_pre, cv.COLOR_BGR2GRAY)

# initial hsv
hsv = np.zeros_like(frame_pre[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]])  
hsv[...,1] = 255 #saturation is full
blank = np.zeros_like(frame_pre)

# video loop
count = 0
print("begin stream!")
print("video resolution is (height, width, channel) : ",frame_pre.shape)
while(True):
# read current frame from cap
    frame_cur = stream.read()
    bgr_cur = cv.cvtColor(frame_cur, cv.COLOR_BGR2GRAY) # change in to gray

    roi_bgr_pre = bgr_pre[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]]
    roi_bgr_cur = bgr_cur[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]]

    # calculate flow
    flow = cv.calcOpticalFlowFarneback(roi_bgr_pre,roi_bgr_cur, None,
                                    pyr_scale,
                                    levels,
                                    winsize,
                                    iterations,
                                    poly_n,
                                    poly_sigma,
                                    flags)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])  # orginal flow
  
    mag_mean = cv.mean(mag)[0]
    mag_sft = abs(mag - mag_mean)  # shifted magnitude to elimiate noise
    hsv[...,0] = (ang + hue)*180/np.pi/2 # color space related to angle 
    hsv[...,2] = cv.normalize(mag_sft,None,0,255,cv.NORM_MINMAX) 
    
    #enhanced flow
    bgr_flow = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    mag_enhanced = zeros_like(mag_sft)
    cv.min(mag_sft,mag_ceiling,mag_enhanced)  # enhance flow, ceiling and flooring
    cv.max(mag_enhanced,mag_floor,mag_enhanced)
    hsv[...,0] = (ang + hue)*180/np.pi/2 # color space related to angle 
    hsv[...,2] = cv.normalize(mag_enhanced,None,0,255,cv.NORM_MINMAX)
    bgr_flow_enhanced = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    
    # image emerge with enhanced flow
    flow_blend_enhance = cv.addWeighted(frame_cur[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]], 1-alpha ,bgr_flow_enhanced,  alpha, 0)
    bgr_blend = cv.addWeighted(frame_cur[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]], 1-alpha ,bgr_flow,  alpha, 0)
    frame_blend = cv.addWeighted(frame_cur, 1-alpha ,blank,  alpha, 0)#frame_cur.copy()
    frame_blend[roi_rect[2]:roi_rect[3],roi_rect[0]:roi_rect[1]] = flow_blend_enhance
    
    # if ture, compare image is  successive; if false, compare image is not change
    if is_ref_refresh:
      bgr_pre = bgr_cur
  
    #palette
    palette=np.zeros((512,512,3),np.uint8)
    mag_norm = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    mag_min,mag_max,min_indx,max_indx=cv.minMaxLoc(mag)
    ang_mean = cv.mean(ang*mag/mag_mean)[0]
    for k in range(600):
        i = random.randrange(0,len(mag))
        j = random.randrange(0,len(mag[0]))
        angle = ang[i][j]
        magnitude = abs(mag[i][j]*100)
        if(magnitude<10000 and mag_mean < 10000):
            x = int(magnitude*math.cos(angle))+256
            y = int(magnitude*math.sin(angle))+256
            cv.circle(palette,(x,y),int(magnitude/10),hsv2rgb((angle+math.pi)/math.pi/2, mag_norm[i][j]/255,1),-1)
    cv.circle(palette,(256,256),int(1.0*100),hsv2rgb((angle+math.pi)/math.pi/2, mag_norm[i][j]/255,1),3)
    cv.circle(palette,(256,256),int(mag_ceiling*100),hsv2rgb((angle+math.pi)/math.pi/2, mag_norm[i][j]/255,2),1)
    cv.circle(palette,(256,256),int(mag_floor*100),hsv2rgb((angle+math.pi)/math.pi/2, mag_norm[i][j]/255,1),1)
    cv.line(palette,(256,256),(int(100*math.cos(ang_mean))+256, int(100*math.sin(ang_mean))+256),(0,255,0),3)
    cv.line(palette,(256,256),(int(50*math.cos(ang[max_indx[1],max_indx[0]]))+256, int(50*math.sin(ang[max_indx[1],max_indx[0]])) +256 ),(255,0,0),3)
    cv.putText(palette, "max="+str(mag.max()), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv.putText(palette, "mag_mean="+str("%.2f"%mag_mean), (0, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv.putText(palette, "mag_max="+str(mag_ceiling), (0, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv.putText(palette, "mag_min="+str(mag_floor), (0, 120), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv.rectangle(frame_blend, (roi_rect[0], roi_rect[2]), (roi_rect[1], roi_rect[3]), (0,255,0), 2)
    
    count = count + 1
 
    # result flow image
    cv.imshow('Orignal',cv.resize(frame_cur, (result_RES[0], result_RES[1])))
    cv.imshow('Area of Intrest Blended', cv.resize(frame_blend  , (result_RES[0], result_RES[1])))
    cv.imshow("Palette", palette)
    cv.imshow('Blender', cv.resize(bgr_blend , (result_RES[0], result_RES[1])))
    #cv.imshow('Blender_enhanced', bgr_blend_enhance)
    #cv.imshow('Flow', cv.resize(bgr_flow , (result_RES[0], result_RES[1])))
    
    
    
    import os
    # if input key 'r', refresh compare image
    key = cv.waitKey(10)
    if key==114:
        #打印时间戳保存
        timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(timeNow)
        
        bgr_pre = bgr_cur
        print("the image is refreshed")
        cv.imwrite(timeNow +'/Orignal.jpg',frame_cur)
        cv.imwrite(timeNow +'/Area_of_Intrest_Blended.jpg', frame_blend)
        cv.imwrite(timeNow +"/Palette.jpg", palette)
        cv.imwrite(timeNow +'/Blender.jpg', bgr_blend)
    # if input key 's', refresh compare image
    # if key==115:
        # 绘制magnitude图片
        print("mag.shape: ", np.shape(mag))
        fig = plt.figure()
        ax3 = plt.axes(projection='3d')
        # 大小需随机定制
        xx = np.arange(0, 1720, 1)
        yy = np.arange(0, 1080, 1)
        X, Y =np.meshgrid(xx, yy)
        Z = np.sin(X)+np.cos(Y)
        print("Z.shape: ", np.shape(Z))
        ax3.plot_surface(X,Y,mag, cmap='rainbow')
        #ax3.plot_surface(X, Y, mag, cmap='rainbow')
        plt.show()
        # 保存信息
        timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(timeNow)
        np.savetxt(timeNow +"/mag.txt", mag)
        np.savetxt(timeNow +"/angle.txt", ang)
    # 绘制magnitude图片
    print("mag.shape: ", np.shape(mag))
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    # 大小需随机定制
    xx = np.arange(0, 1720, 1)
    yy = np.arange(0, 1080, 1)
    X, Y =np.meshgrid(xx, yy)
    Z = np.sin(X)+np.cos(Y)
    print("Z.shape: ", np.shape(Z))
    ax3.plot_surface(X,Y,mag, cmap='rainbow')
    #ax3.plot_surface(X, Y, mag, cmap='rainbow')
    plt.show()
    # 保存信息
    timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(timeNow)
    np.savetxt(timeNow +"/mag.txt", mag)
    np.savetxt(timeNow +"/angle.txt", ang)
cap.release()
save_orginal.release()
save_roi_enhanced.release()
cv.destroyAllWindows()

