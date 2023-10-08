import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import cupy as cp

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
    norm_type = cv2.NORM_MINMAX
    if half == "True":
        depth_map = depth_map.astype(cp.float32)
        norm_type = cv2.NORM_MINMAX
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=norm_type)
        depth_map = depth_map.astype(cp.uint8)
    else:
        depth_map = cv2.normalize(depth_map, None, 255, 0, norm_type=norm_type, dtype=cv2.CV_8U)

    return depth_map

def depth_extract_video(video_file, output_path, width, height, model, transform, device, half, deflicker):
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    prev_frame = None

    start_time = time.time()
    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        depth_frame = depth_extract(half, frame, model, transform, device)

        if deflicker == "True" and prev_frame is not None:
            depth_frame = deflicker_depth_frame(depth_frame, prev_frame, half)

        prev_frame = depth_frame

        out.write(depth_frame)
        cv2.imshow('Depth, press CTRL + Z inside terminal to close', depth_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        current_fps = (i + 1) / elapsed_time
        print(f"Current FPS: {round(current_fps, 2)}", end="\r")

    elapsed_time = time.time() - start_time
    average_fps = frame_count / elapsed_time
    print(f"Average FPS: {round(average_fps, 2)}")

    video.release()
    out.release()
    cv2.destroyAllWindows()