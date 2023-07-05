import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
import cupy as cp
import threading

def deflicker_depth_frame(frame, prev_frame, half):
    # Poor mans approach to deflickering
    if half:
        frame = cp.asarray(frame, dtype=cp.float16)
        prev_frame = cp.asarray(prev_frame, dtype=cp.float16)
    else:
        frame = np.asarray(frame, dtype=np.float32)
        prev_frame = np.asarray(prev_frame, dtype=np.float32)

    diff = frame - prev_frame
    flicker_magnitude = np.abs(diff).mean()
    deflickering_factor = 1 * (flicker_magnitude / 255)
    deflickered_frame = frame - deflickering_factor * diff
    deflickered_frame = np.clip(deflickered_frame, 0, 255)

    if half:
        deflickered_frame = cp.asnumpy(deflickered_frame)

    return deflickered_frame

def depth_extract(half, frame, model, transform, device):
    if half == "True":
        frame = Image.fromarray(frame)
        img = transform(frame).unsqueeze(0)
        img = img.to(device, dtype=torch.float16)

        with torch.no_grad():
            prediction = model(img)

        depth_map = prediction[0].cpu().numpy()

        depth_map = depth_map.astype(cp.float32)
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        depth_map = depth_map.astype(cp.uint8)
    else:
        frame = Image.fromarray(frame)
        img = transform(frame).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            prediction = model(img)

        depth_map = prediction[0].cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return depth_map

def build_buffer(video_file, buffer, width, height):
    video = cv2.VideoCapture(video_file)

    if not video.isOpened():
        print("Error opening video file")
        return

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        buffer.append(frame)
    
    video.release()
    print(f"\rVideo buffering thread is finished")
    
def depth_extract_video(buffer, video_file, output_path, width, height, model, transform, device, half, deflicker, video, fps, fourcc, out):

    if len(buffer) == 0:
        print("No frames in buffer")
        return

    prev_frame = None

    for frame in buffer:
        
        time_start = time.time()
        depth_frame = depth_extract(half, frame, model, transform, device)
        if deflicker == "True" and prev_frame is not None:
            depth_frame = deflicker_depth_frame(depth_frame, prev_frame, half)
            prev_frame = depth_frame
        else:
            prev_frame = depth_frame

        if depth_frame.dtype != np.uint8:
            depth_frame = depth_frame.astype(np.uint8)
            
        out.write(depth_frame)

        #cv2.imshow('Input, press Q to close', frame)
        cv2.imshow('Depth, press Q to close',depth_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            break
        
        if time.time() - time_start > 0:
            instant_fps = 1 / (time.time() - time_start)
        else:
            instant_fps = 0
            
        alpha = 0.1
        fps = (1 - alpha) * fps + alpha * instant_fps
        print(f"\rFPS: {round(fps, 2)}", end="")
        
    out.release()
    cv2.destroyAllWindows()

def depth_extract_video_with_vfi(video_file, output_path, width, height, model, transform, device, half, deflicker):
    video = cv2.VideoCapture(video_file)
    video_copy = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_width, frame_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    if not video.isOpened():
        print("Error opening video file")
        return

    prev_frame = None
    
    i = 0
    
    for i in range(frame_count):
        time_start = time.time()
        ret, frame = video.read(i)
        frame = cv2.resize(frame, (width, height))
        if not ret:
            break
        
        if i % 2 == 0:
            depth_frame = depth_extract(half, frame, model, transform, device)
            if deflicker == "True" and prev_frame is not None:
                depth_frame = deflicker_depth_frame(depth_frame, prev_frame, half)
                prev_frame = depth_frame
            else:
                prev_frame = depth_frame
            if depth_frame.dtype == np.uint8:
                out.write(depth_frame)
            else:
                depth_frame = depth_frame.astype(np.uint8)
                out.write(depth_frame)
        else:
            continue
        
        cv2.imshow('Input, press Q to close', frame)
        cv2.imshow('Depth, press Q to close', depth_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            break
        
        if time.time() - time_start > 0:
            instant_fps = 1 / (time.time() - time_start)
        else:
            instant_fps = 0
            
        alpha = 0.1
        fps = (1 - alpha) * fps + alpha * instant_fps
        print(f"\rFPS: {round(fps, 2)}", end="")
        

    video.release()
    cv2.destroyAllWindows()

def video_threading(video_file, output_path, width, height, model, transform, skip, device, half, deflicker, nt):
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    buffer = []
    buffer_thread = threading.Thread(target=build_buffer, args=(video_file, buffer, width, height))
    
    
    if skip == "True":
        processing_threads = [
            threading.Thread(target=depth_extract_video_with_vfi, args=(buffer, video_file, output_path, width, height, model, transform, device, half, deflicker, video, fps, fourcc, out))
            for _ in range(nt)
        ]
    else:
        processing_threads = [
            threading.Thread(target=depth_extract_video, args=(buffer, video_file, output_path, width, height, model, transform, device, half, deflicker, video, fps, fourcc, out))
            for _ in range(nt)
        ]
        
    
    
    buffer_thread.start()

    # Wait a second for the buffer to build up some frames
    time.sleep(1)

    for thread in processing_threads:
        thread.start()

    buffer_thread.join()
    
    for thread in processing_threads:
        thread.join()
