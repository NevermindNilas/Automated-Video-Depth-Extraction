import cv2 
import time 
from threading import Thread
from globals import decode_buffer, write_buffer, deflicker_buffer
import numpy as np

#from decord import VideoReader
#from decord import cpu

# Yoinked and then modified from:
# https://github.com/vasugupta9/DeepLearningProjects/blob/main/MultiThreadedVideoProcessing/video_processing_parallel.py 


class VideoDecodeStream:
    def __init__(self, video_file):
        self.vcap = cv2.VideoCapture(video_file)

        # initialize a frame in order to not store a none value in the buffer
        self.grabbed , self.frame = self.vcap.read()
        decode_buffer.append(self.frame)

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

    I still can't quite figure out why this doesn't work.
    Work  in progress
    '''
    def __init__(self, output_file, fourcc, fps, width, height):
        self.output_file = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)
        self.thread = Thread(target=self.update , args=())
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
    
    def update(self):
        time.sleep(1)
        while True:
            if len(write_buffer) > 0:
                self.output_file.write(write_buffer.pop(0))
            if len(write_buffer) == 0 and len(decode_buffer) == 0:
                break
        self.output_file.release()
        
    def writer(self, frame):
        write_buffer.append(frame)

class DepthScanning:
    '''
    Do the depth scan on nt threads
    
    Work  in progress
    '''
    def __init__(self, model, device, half, nt):
        self.model = model
        self.device = device
        self.half = half
    
class Deflicker:
    """
    Delficker using a median filter
    """
   
    def __init__(self):
       self.thread = Thread(target=self.update , args=())
       self.thread.daemon = True

    def start(self):
        self.thread.start()
    
    def update(self):
        time.sleep(3) # wait a few seconds for some frames to build up
        while True:
            if len(deflicker_buffer) > 0:
                frame = deflicker_buffer.pop(0)
                # Convert the frame to grayscale
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Compute the median of the grayscale frame using a 3x3 kernel
                median = cv2.medianBlur(frame, 3)
                # Compute the difference between the median and the original frame
                diff = cv2.absdiff(frame, median)
                # Compute the thresholded difference to remove small changes
                thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
                # Apply a morphological closing operation to fill in gaps
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                # Apply the thresholded difference to the original frame to remove flicker
                deflickered = cv2.bitwise_and(frame, frame, mask=thresh)
                write_buffer.append(deflickered)
            
            if len(deflicker_buffer) == 0 and len(decode_buffer) == 0:
                break
        