import torch
import cv2
import numpy as np
import concurrent.futures
import time
from parallel_processing import VideoDecodeStream
import asyncio

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

def depth_extract_video(video_file, output_path, width, height, model, device, half, nt, verbose):
    video = VideoDecodeStream(video_file, width, height)
    fps = video.get_fps()
    fourcc = video.fourcc()
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    video.start()
    start_time = time.time()

    async def update_fps(current_fps, i, start_time):
        elapsed_time = time.time() - start_time
        current_fps = (i + 1) / elapsed_time
        print(f"Current FPS: {round(current_fps, 2)}", end="\r")
        return current_fps
    
    async def main():
        loop = asyncio.get_running_loop()
        current_fps = 0
        i = 0
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=nt) as executor:
            futures = []
            while True:
                frame = video.read()
                if frame is None and video.stopped is True:
                    break
                
                future = executor.submit(depth_extract, half, frame, model, device)
                futures.append(future)
                if len(futures) >= nt:
                    depth_frames = [f.result() for f in futures]
                    for depth_frame in depth_frames:
                        out.write(depth_frame)
                        if verbose == "True": # Imshow actually has a 10% performance hit so if you don't need it, don't use it
                            cv2.imshow('Depth, press CTRL + C inside the terminal to exit', depth_frame)
                    futures = []
                    
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                
                i += 1
                task = loop.create_task(update_fps(current_fps, i, start_time))
                current_fps = await task
                
        video.stop()
        out.release()
        cv2.destroyAllWindows()
        
    asyncio.run(main())
            

    total_process_time = time.time() - start_time
    print(f"Process done in {total_process_time:.2f} seconds")
    video.stop()
    
    '''    
    while True:
        frame = video.read()
        if frame is None and video.stopped is True:
            break
        depth_frame = depth_extract(half, frame, model, device)
        out.write(depth_frame)       
        cv2.imshow('Depth, press CTRL + C inside the terminal to exit', depth_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break     
            
        elapsed_time = time.time() - start_time
        current_fps = (i + 1) / elapsed_time
        print(f"Current FPS: {round(current_fps, 2)}", end="\r") 
        i += 1'''
                