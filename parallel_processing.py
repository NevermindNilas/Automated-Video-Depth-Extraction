# importing required libraries 
import cv2 
import time 
from threading import Thread # library for implementing multi-threaded processing 
# Yoinked and then modified from:
# https://github.com/vasugupta9/DeepLearningProjects/blob/main/MultiThreadedVideoProcessing/video_processing_parallel.py 

global decode_buffer
decode_buffer = []

class VideoDecodeStream:
    def __init__(self, video_file):
        self.vcap = cv2.VideoCapture(video_file)
            
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = True 

        # initialize a frame in order to not store a none value in the buffer
        self.grabbed , self.frame = self.vcap.read()
        decode_buffer.append(self.frame)
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True 

    def start(self):
        self.stopped = False
        self.t.start() 

    def update(self):
        while True :
            if self.stopped is True :
                break
            grabbed, frame = self.vcap.read(cv2.IMREAD_GRAYSCALE)
            
            if grabbed is False :
                print('The video buffering has been finished')
                self.stopped = True
                break
            
            decode_buffer.append(frame)
        self.vcap.release()

    def read(self):
        if decode_buffer:
            frame = decode_buffer.pop(0)
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
 
class VideoWriteStream:
    '''
    Encode the depth scan frames as soon as they are ready and parallelize the process
    
    Work  in progress
    '''
    def __init__(self, output_file, fourcc, fps, width, height):
        self.output_file = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        self.t = Thread(target=self.update , args=())
        self.t.daemon = True
        
    def start(self):
        self.stopped = False
        self.t.start()
    
    def update(self):
        while True:
            if self.stopped is True:
                break

class DepthScanning:
    '''
    Do the depth scan on nt threads
    
    Work  in progress
    '''
    def __init__(self, model, device, half, nt):
        self.model = model
        self.device = device
        self.half = half