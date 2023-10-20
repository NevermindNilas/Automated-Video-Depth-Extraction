import cv2 
import time 
from threading import Thread
import numpy as np
#import torch

#from decord import VideoReader
#from decord import cpu

# Yoinked and then modified from:
# https://github.com/vasugupta9/DeepLearningProjects/blob/main/MultiThreadedVideoProcessing/video_processing_parallel.py 

class VideoDecodeStream:
    def __init__(self, video_file, width, height):
        self.vcap = cv2.VideoCapture(video_file)
        self.width = width
        self.height = height
        self.decode_buffer = []
        # initialize a frame in order to not store a none value in the buffer
        self.grabbed , self.frame = self.vcap.read()
        self.frame = cv2.resize(self.frame, (self.width, self.height))
        self.decode_buffer.append(self.frame)
        
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True 

    def start(self):
        self.stopped = False
        self.t.start() 

    def update(self):
        while True :
            if self.stopped is True :
                break
            
            grabbed, frame = self.vcap.read()
            if grabbed is False :
                print('The video buffering has been finished')
                self.stopped = True
                break
            
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.decode_buffer.append(frame)
        self.vcap.release()

    def read(self):
        if self.decode_buffer:
            frame = self.decode_buffer.pop(0)
            return frame

    def stop(self):
        self.stopped = True 
    
    def get_fps(self):
        self.fps = self.vcap.get(cv2.CAP_PROP_FPS)
        return self.fps
    
    def get_frame_count(self):
        self.frame_count = self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        return self.frame_count
    
    def fourcc(self):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return self.fourcc
 
