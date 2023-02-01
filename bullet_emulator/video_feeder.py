from stat import SF_IMMUTABLE
import cv2
import time
import sys
import numpy as np
import os

def rescale_frame(frame, percent=3/4):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_NEAREST), width


def video_to_jpeg(file_path = '4k_video/downtown_miami.mp4', scaling_factor = 1, jpeg_path = './jpg_result'):

    # raw video file
    #file_name = 'downtown_miami.mp4'
    #cap = cv2.VideoCapture('4k_video/' + file_name)
    cap = cv2.VideoCapture(file_path)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    i=0
    sf = scaling_factor
    ret, frame = cap.read()
    width = int(frame.shape[1] * sf)
    height = int(frame.shape[0] * sf)

    # configure save path
    path = './jpg_result'
    sf_path = './jpg_result/' + file_name + '_' + str(height) +'p'
    if not os.path.exists(path):
        os.makedirs(path)
        print('Created file folder')
    if not os.path.exists(sf_path):
        os.mkdir(sf_path)
        print('Created sf file folder')

    print('Frame size: ', frame.shape[0], frame.shape[1])
    print('Reshaped frame size: ', height, width)

    while(True):
        try:
            ret, frame = cap.read()
        except:
            print('No remaining frame')
            break
        frame, width = rescale_frame(frame, sf)
        filename='frame_'+str(i)
        path = sf_path+'/'+filename+'.jpg'
        if ret:
            img = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY,50])
            file_size = os.path.getsize(path)
            print(path + ' ' + str(round(file_size/(1024),2))+' KB',  end='\r', flush=True)
            # np.save(path, frame) # x_save.npy
        
        # The below code shows the current encoded frames
        # cv2.imshow('test', frame)
        # cv2.waitKey(1)
        i=i+1