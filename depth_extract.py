import torch
import cv2
import numpy as np
import time
from threading import Thread

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
                print('The video buffering has been finished')
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
    def __init__(self, model_type, half, video_stream):
        self.depth_buffer = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).to(self.device)
        self.model.eval()
        self.half = half
        self.video_stream = video_stream
        
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
            return depth_map
        
    def update(self):
        while not self.stopped:
            image = self.video_stream.read()
            if image is not None:
                depth_frame = self.depth_extract(image)
                self.depth_buffer.append(depth_frame)
            elif self.video_stream.stopped == True:
                print('The video depth scanning has been finished')
                self.stopped = True
                break
    
    def read(self):
        return self.depth_buffer.pop(0) if self.depth_buffer else None

def depth_extract_video(video_file, output_path, width, height, half, nt, verbose, model_type):
    video_stream = VideoDecodeStream(video_file, width, height)
    video_stream.start()
    depth_stream = DepthScanStream(model_type, half, video_stream)
    depth_stream.start()
    
    out = cv2.VideoWriter(output_path, video_stream.get_fourcc(), video_stream.get_fps(), (width, height), isColor=False)
    start_time = time.time()

    i = 0 # for fps calculation
    while True:
        depth_frame = depth_stream.read()
        if depth_frame is None and depth_stream.stopped == True:
            break
        elif verbose == "True" and depth_frame is not None:
            cv2.imshow('Depth, press CTRL + C inside the terminal to exit', depth_frame)
            i += 1
            elapsed_time = time.time() - start_time
            current_fps = (i + 1) / elapsed_time
            print(f"Current FPS: {round(current_fps, 2)}", end="\r")
            
        out.write(depth_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    out.release()
    cv2.destroyAllWindows()
    total_process_time = time.time() - start_time
    print(f"Process done in {total_process_time:.2f} seconds")
