import torch
import cv2
import numpy as np
import time
from threading import Thread
from tqdm import tqdm
from torch.nn import functional as F
from queue import Queue, Empty


class VideoDecodeStream:
    def __init__(self, video_file, width, height):
        self.vcap = cv2.VideoCapture(video_file)
        self.width = width
        self.height = height
        self.decode_buffer = []
        self.grabbed , self.frame = self.vcap.read()
        self.frame = cv2.resize(self.frame, (self.width, self.height))
        self.decode_buffer.append(self.frame)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True 

    def start(self):
        self.stopped = False
        self.thread.start() 

    def update(self):
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                self.stopped = True
                break
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.decode_buffer.append(frame)
        self.vcap.release()

    def read(self):
        return self.decode_buffer.pop(0) if self.decode_buffer else None

    def stop(self):
        self.stopped = True 

    def get_fps(self):
        return self.vcap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_fourcc(self):
        return cv2.VideoWriter_fourcc(*'mp4v')
    
class DepthScanStream:
    def __init__(self,half, video_stream, model, device, progressbar):
        self.depth_buffer = []
        self.half = half
        self.video_stream = video_stream
        self.model = model
        self.device = device
        self.progressbar = progressbar
        # One frame has to be initilized otherwise the for loop will just skip it and imshow will go nutzzzz
        # This can be potentially bad on I/O bound systems, but it's the only way I found to make it work
        frame = self.video_stream.read()
        if frame is not None:
            depth_frame = self.depth_extract(frame)
            self.depth_buffer.append(depth_frame)
            
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()
        
    def stop(self):
        self.stopped = True     
        
    def depth_extract(self, image):
        with torch.no_grad():
            img = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
            if self.half == "True":
                img = img.half()
                self.model = self.model.half()
            img = img.to(self.device)
            prediction = self.model(img)
            depth_map = prediction[0].cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
            depth_map = depth_map.astype(np.uint8)
            self.progressbar.update(1)
            return depth_map

    def update(self):
        while not self.stopped:
            image = self.video_stream.read()
            if image is not None:
                depth_frame = self.depth_extract(image)
                self.depth_buffer.append(depth_frame)
            elif self.video_stream.stopped == True:
                self.stopped = True
                break
    
    def read(self):
        return self.depth_buffer.pop(0) if self.depth_buffer else None
    
def load_model(model_type, half):
    " Had to separate load model from videodepthstream because of new users who had to download the model first could run into issues with timing "
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).to(device)
    #model = torch.hub.load("facebookresearch/dinov2", model="dinov2_vitl14").to(device) WTH is this model?? WHY DO I NEED 2 MODELS TO DO ONE JOB???
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if(half):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model.eval()
    model.cuda()
    
    return model, device

def depth_extract_video(video_file, output_path, width, height, half, nt, verbose, model_type):
    model, device = load_model(model_type, half)
    video_stream = VideoDecodeStream(video_file, width, height)
    video_stream.start()
    progressbar = tqdm(total=video_stream.get_frame_count(), unit="frames")
    depth_stream = DepthScanStream(half, video_stream, model, device, progressbar)
    depth_stream.start()
    out = cv2.VideoWriter(output_path, depth_stream.video_stream.get_fourcc(), depth_stream.video_stream.get_fps(), (width, height))
    while True:
        depth_frame = depth_stream.read()
        if depth_frame is None and depth_stream.stopped == True:
            break
        elif verbose == "True" and depth_frame is not None:
            try:
                cv2.imshow('Depth, press CTRL + C inside the terminal to exit', depth_frame)
            except:
                pass
            
        out.write(depth_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
