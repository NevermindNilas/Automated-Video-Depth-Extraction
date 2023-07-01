import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
    
def depth_extract(half, frame, model, transform, device):
    if half == "True":
        frame = Image.fromarray(frame)
        img = transform(frame).unsqueeze(0)
        

        img = img.to(device, dtype=torch.float16)
        
        with torch.no_grad():
            prediction = model(img)
        
        depth_map = prediction[0].cpu().numpy()
        
        depth_map = depth_map.astype(np.float32)
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
    else:
        # Load the image
        frame = Image.fromarray(frame)
        img = transform(frame).unsqueeze(0)
        img = img.to(device)
        # Run the model
        with torch.no_grad():
            prediction = model(img)

        # Convert the prediction to a numpy array
        depth_map = prediction[0].cpu().numpy()

        # Normalize the depth map
        depth_map = cv2.normalize(depth_map, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map = depth_map.astype(np.uint8)
        
    return depth_map

def depth_extract_video(video_file, output_path, width, height, model, transform, device, skip, half):
    
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # this is input frame width and height
    frame_width, frame_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # I will need to make this container agnostic but OpenCV container naming feels like too big brain for me
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')    
    
    out = cv2.VideoWriter(output_path, fourcc, float(fps),(width,height))
    
    # Just in case someone forgets to set the width and height
    if not (height or width):
        height, width = frame_height, frame_width
    else:
        if (height or width) % 32 != 0:
            sys.exit("Error: width and height must be divisible by 32, padding will be coming soon")
            
    
    if not video.isOpened():
        print("Error opening video file")
        return
    
    # frameskip
    if skip == "False":
        while True:
            time_start = time.time()
            ret, frame = video.read()
            
            if not ret:
                break
            
            frame = cv2.resize(frame,(width, height))
            
            cv2.imshow('Input, press CTRL + Z inside terminal to close', frame)
            frame = depth_extract(half, frame, model, transform, device)
            cv2.imshow('Depth', frame)
            
            out.write(frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            alpha = 0.01
            if time.time()-time_start > 0:
                fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                time_start = time.time()
            print(f"\rFPS: {round(fps,2)}", end="")
    else:
        i = 1
        while True:
            time_start = time.time()
            ret,frame = video.read()
            
            if not ret:
                break
            
            frame = cv2.resize(frame,(width, height))
            
            if i % 2 != 0:
                cv2.imshow('Input', frame)
                depth_frame = depth_extract(half, frame, model, transform, device)
                cv2.imshow('Depth', depth_frame)
                out.write(depth_frame)
            else:
                cv2.imshow('Input', frame)
                cv2.imshow('Depth', depth_frame)
                #cv2.imshow('Depth', frame)
                #frame = Rife
            
            i += 1
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            alpha = 0.01
            if time.time()-time_start > 0:
                fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                time_start = time.time()
            print(f"\rFPS: {round(fps,2)}", end="")
            
          
    video.release()
    cv2.destroyAllWindows()