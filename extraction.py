import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import time

# Load the model
# I would probably move and rename this def to a different file in order to keep this cleaner
def depth(model_type, width, height):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    '''
    Debugging
    entrypoints = torch.hub.list("intel-isl/MiDaS", force_reload=True)
    print(entrypoints)
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    '''
    
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)                       
    if device.type == 'cuda':
        model = model.to(device)

    model.eval()

        # Transform the input
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),  # Resize to dimensions divisible by 32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )    
    return device, model, transform

def depth_extract(frame, model, transform, device):
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

def depth_extract_video(video_file, output_path, width, height, model, transform, device):
    
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    # I will need to make this container agnostic but OpenCV container naming feels like too big brain for me
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    if not video.isOpened():
        print("Error opening video file")
        return
    
    while True:
        time_start = time.time()
        ret, frame = video.read()
        frame = cv2.resize(frame,(width, height))
        
        if not ret:
            break
        
        cv2.imshow('Input', frame)
        frame = depth_extract(frame, model, transform, device)
        cv2.imshow('Depth', frame)
        out.write(frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        alpha = 0.01
        if time.time()-time_start > 0:
            fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
            time_start = time.time()
        print(f"\rFPS: {round(fps,2)}", end="")
    
    video.release()
    cv2.destroyAllWindows()

def main(output, model_type, width, height):
    
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")
    
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get the torch properties. Maybe declaring it globally was better in terms of startup times
    device, model, transform = depth(model_type, width, height)
    
    video_files = [
        f for f in os.listdir(input_path) if f.endswith(
            ('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()
    
    for video_file in video_files:
        video_file = os.path.join(input_path, video_file)
        output_path = os.path.join(output_path, output)
        print("Processing Video File:", video_file)
        depth_extract_video(video_file, output_path, width, height, model, transform, device)
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")

    parser.add_argument(
        "-width",
        type=int,
        help="Width of the corresponding output, must be a multiple of 32",
        default=None,
        action="store"
    )

    parser.add_argument(
        "-height",
        type=int,
        help="Height of the corresponding output, must be a multiple of 32",
        default=None,
        action="store"
    )
    
    # No idea how and why custom isn't functional also no idea why I can't use 3.1 models like SWIN2_Large, help!!!!
    parser.add_argument(
        "-model_type",
        type=str,
        help="Which MIDAS model to choose from, e.g DPT_Large, DPT_Hybrid or path/to/model, custom isn't functional for now.",
        default="DPT_Hybrid",
            action="store"
        )
    
    parser.add_argument(
        "-output", 
        type=str,
        help="The output's path",
        default=None,
        action="store"
    )
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.output, args.model_type, args.width, args.height)