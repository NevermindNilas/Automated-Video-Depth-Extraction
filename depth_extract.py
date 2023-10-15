import torch
import cv2
import numpy as np
from PIL import Image
import time
from parallel_processing import VideoDecodeStream, VideoWriteStream

def depth_extract(half, frame, model, device):
    with torch.no_grad():
        img = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)
        if half == "True":
            img = img.half()
            model = model.half()
        img = img.to(device)
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

def depth_extract_video(video_file, output_path, width, height, model, device, half):
    video = VideoDecodeStream(video_file)
    fps = video.get_fps()
    fourcc = video.fourcc()
    #frame_count = int(video.get_frame_count())
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    video.start()
    start_time = time.time()    
    i = 0

    while True:
        frame = video.read()
        if frame is None and video.stopped is True:
            break
        
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)    
        depth_frame = depth_extract(half, frame, model, device)
        
        out.write(depth_frame)
        cv2.imshow('Depth, press CTRL + C inside the terminal to exit', depth_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break     
            
        elapsed_time = time.time() - start_time
        current_fps = (i + 1) / elapsed_time
        print(f"Current FPS: {round(current_fps, 2)}", end="\r") 
        i += 1

    total_process_time = time.time() - start_time
    print (f"Process done in {total_process_time:.2f} seconds") 
    video.stop()
    out.release()
