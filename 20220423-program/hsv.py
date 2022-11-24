#hsv transform to rgb format
def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

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

# 处理mag ang函数
mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])  # orginal flow
mag = mag[para['Noffset']:mag.shape[0]-para['Noffset'], para['Noffset']:(mag.shape[1]-para['Noffset'])]
ang = ang[para['Noffset']:ang.shape[0]-para['Noffset'], para['Noffset']:(ang.shape[1]-para['Noffset'])]

mag_mean = cv.mean(mag)[0]
mag_sft = abs(mag - mag_mean)  # shifted magnitude to elimiate noise
mag_enhanced = zeros_like(mag_sft)
cv.min(mag_sft,para['mag_ceiling'],mag_enhanced)  # enhance flow, ceiling and flooring
cv.max(mag_enhanced,para['mag_floor'],mag_enhanced)

hsv[...,0] = (ang + para['hue'])*180/np.pi/2 # color space related to angle 
hsv[...,2] = cv.normalize(mag_enhanced,None,0,255,cv.NORM_MINMAX)
bgr_flow_enhanced = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

# image emerge with enhanced flow
flow_blend_enhance = cv.addWeighted(frame_cur[solution['roi_rect'][2]+para['Noffset']:solution['roi_rect'][3]-para['Noffset'],solution['roi_rect'][0]+para['Noffset']:solution['roi_rect'][1]-para['Noffset']], 1-para['alpha'] ,bgr_flow_enhanced,  para['alpha'], 0)
frame_blend = cv.addWeighted(frame_cur, 1-para['alpha'] ,blank,  para['alpha'], 0)#frame_cur.copy()
frame_blend[solution['roi_rect'][2]+para['Noffset']:solution['roi_rect'][3]-para['Noffset'],solution['roi_rect'][0]+para['Noffset']:solution['roi_rect'][1]-para['Noffset']] = flow_blend_enhance

# cv.imshow('Blender', cv.resize(bgr_blend , (para['result_RES'][0], para['result_RES'][1])))     
cv.imshow('Area of Intrest Blended', cv.resize(frame_blend  , (para['result_RES'][0], para['result_RES'][1])))
