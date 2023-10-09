import torch
import cv2
import numpy as np
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor

def depth_extract(half, frame, model, transform, device):
    frame = Image.fromarray(frame)
    img = transform(frame).unsqueeze(0)
    dtype = torch.float16 if half == "True" else torch.float32
    img = img.to(device, dtype=dtype)

    with torch.no_grad():
        prediction = model(img)

    depth_map = prediction[0].cpu().numpy()
    if half == "True":
        depth_map = depth_map.astype(np.float32)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)
    else:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)

    return depth_map

def depth_extract_video(video_file, output_path, width, height, model, transform, device, half, nt):
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    start_time = time.time()
    
    with ThreadPoolExecutor(nt) as executor:
        futures = []
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            future = executor.submit(depth_extract, half, frame, model, transform, device)
        
            depth_frame = future.result()
            out.write(depth_frame)
            cv2.imshow('Depth, press CTRL + C inside terminal to close', depth_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            elapsed_time = time.time() - start_time
            current_fps = (i + 1) / elapsed_time
            print(f"Current FPS: {round(current_fps, 2)}", end="\r")
            
    print ("Process done in", time.time() - start_time, "seconds")