import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import cupy as cp
import os

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
    deflickering_factor = 0.5 * (flicker_magnitude / 255)
    deflickered_frame = frame - deflickering_factor * diff
    deflickered_frame = np.clip(deflickered_frame, 0, 255)

    if half:
        deflickered_frame = cp.asnumpy(deflickered_frame)
        
    return deflickered_frame

def depth_extract(half, frame, model, transform, device):
    frame = Image.fromarray(frame)
    img = transform(frame).unsqueeze(0)
    dtype = torch.float16 if half == "True" else torch.float32
    img = img.to(device, dtype=dtype)

    with torch.no_grad():
        prediction = model(img)

    depth_map = prediction[0].cpu().numpy()
    if half == "True":
        depth_map = depth_map.astype(np.float16)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)
    else:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)

    return depth_map

def depth_extract_video(video_file, output_path, width, height, model, transform, device, half):
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    start_time = time.time()
    
    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_frame = depth_extract(half, frame, model, transform, device)

        out.write(depth_frame)
        cv2.imshow('Depth, press CTRL + C inside terminal to close', depth_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        current_fps = (i + 1) / elapsed_time
        print(f"Current FPS: {round(current_fps, 2)}", end="\r")

    video.release()
    out.release()
    cv2.destroyAllWindows()

def depth_extract_video_deflicker(video_file, output_path, width, height, model, transform, device, half):
    print("Deflickering is enabled, this will reduce the FPS")
    
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    output_folder = os.path.join(output_path, "temp_folder")

    depth_frames = []
    start_time = time.time()
    
    #frame_count = 25
    temp_folder = os.path.join(output_folder, "temp_folder")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        
    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_frame = depth_extract(half, frame, model, transform, device)
        
        depth_frames.append(depth_frame)
        
        cv2.imshow('Depth, press CTRL + C inside terminal to close', depth_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        current_fps = (i + 1) / elapsed_time
        print(f"Current FPS: {round(current_fps, 2)}", end="\r")

    video.release()
    cv2.destroyAllWindows()  
    
    for i, depth_frame in enumerate(depth_frames):
        filename = os.path.join(temp_folder, f"{i:06d}.png")
        cv2.imwrite(filename, depth_frame)
    
'''    
    print(f"Depth frames written to {temp_folder}")
    
    brightness = calc_brightness(depth_frames, temp_folder)
    
def calc_brightness(depth_frames, sigma=2.5):
    brightness = []
    for depth_frame in enumerate(depth_frames):
        mask = np.ones_like(depth_frame[:, :, 0], dtype=bool)

        if sigma is not None:
            for channel in range(3):
                mean = np.mean(depth_frame[:, :, channel])
                std = np.std(depth_frame[:, :, channel])
                dist = np.abs(depth_frame[:, :, channel] - mean) / std
                mask[dist > sigma] = False

        brightness.append(np.mean(depth_frame[mask]))

    return np.array(brightness)
'''