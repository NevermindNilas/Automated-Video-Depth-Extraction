import cv2 
import time 
from threading import Thread
from globals import decode_buffer, write_buffer, deflicker_buffer, video_player_buffer, depth_scan_buffer
import numpy as np
import torch

#from decord import VideoReader
#from decord import cpu

# Yoinked and then modified from:
# https://github.com/vasugupta9/DeepLearningProjects/blob/main/MultiThreadedVideoProcessing/video_processing_parallel.py 


class VideoDecodeStream:
    def __init__(self, video_file, width, height):
        self.vcap = cv2.VideoCapture(video_file)
        self.width = width
        self.height = height

        # initialize a frame in order to not store a none value in the buffer
        self.grabbed , self.frame = self.vcap.read()
        self.frame = cv2.resize(self.frame, (self.width, self.height))
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
            
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
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

'''class DepthScanning:

    #Do the depth scan on nt threads
    
    #Work  in progress

    def __init__(self, model, device, half, nt, width, height):
        self.model = model
        self.device = device
        self.half = half
        self.nt = nt
        self.width = width
        
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True 
        
    def start(self):
        self.thread.start()
    
    def udpate(self):
        while True:
            if len(decode_buffer) > 0:
                frame = decode_buffer.pop(0)
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                with torch.no_grad():
                    img = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)
                    if self.half == "True":
                        img = img.half()
                        model = model.half()
                    img = img.to(self.device)
                    prediction = model(img)
                    depth_map = prediction[0].cpu().numpy()
                    if self.half == "True":
                        depth_map = depth_map.astype(np.float32)
                        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                        depth_map = depth_map.astype(np.uint8)
                    else:
                        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                        depth_map = depth_map.astype(np.uint8)
                    depth_scan_buffer.append(depth_map)
                
                if len(decode_buffer) == 0 and len(depth_scan_buffer) == 0:
                    break
    
class VideoPlayerStream:

    #Video player buffer
    #Work in progress

    def __init__(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
    
    def start(self):
        self.thread.start()
    
    def update(self):
        time.sleep(1) # wait a second for some frames to build up
        while True:
            if len(video_player_buffer) > 0:
                frame = video_player_buffer.pop(0)
                cv2.imshow('Depth, press CTRL + C inside the terminal to exit', frame)
            if len(video_player_buffer) == 0 and len(decode_buffer) == 0:
                break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
class Deflicker:

    #Delficker using a median filter

   
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

'''