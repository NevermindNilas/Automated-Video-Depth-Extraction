import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import sys
from depth_extract import depth_extract_video

def load_device(half, model_type, width, height):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    if half == "True":
        print('Using half precision')
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = torch.hub.load("intel-isl/MiDaS", model_type).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return device, model, transform

def main(deflicker, half, model_type, height, width):
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Check if the width and height are multiples of 32
    if width % 32 != 0:
        print("The width is not divisible by 32, it will be rounded up to the nearest multiple of 32")
        width = (width // 32 + 1) * 32
    
    if height % 32 != 0:
        print("The height is not divisible by 32, it will be rounded up to the nearest multiple of 32")
        height = (height // 32 + 1) * 32

    device, model, transform = load_device(half, model_type, width, height)

    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()

    if not video_files:
        sys.exit("No videos found in the input folder")
    
    for i, video_file in enumerate(video_files):
        output = video_file
        if output.endswith(".mp4"):
            output = output.split('.')[0] + "_" + str(i) + ".mp4"
        elif output.endswith((".mov", ".avi", ".mkv")):
            print("The only accepted container for output is mp4")
            output = output.split('.')[0] + str(i) + ".mp4"
        else:
            output = output + "_" + str(i) + ".mp4"

        video_file = os.path.join(input_path, video_file)
        output_path = os.path.join(output_path, output)
        print("Processing Video File:", video_file)
        depth_extract_video(video_file, output_path, width, height, model, transform, device, half, deflicker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")

    parser.add_argument('-width', type=int, help="Width of the corresponding output, must be a multiple of 32", default=None)
    parser.add_argument("-height", type=int, help="Height of the corresponding output, must be a multiple of 32", default=None)
    parser.add_argument('-model_type', required=False, type=str, help="Which MIDAS model to choose from, e.g DPT_Large, DPT_Hybrid or path/to/model, custom isn't functional for now.", default="DPT_Hybrid", action="store")
    parser.add_argument('-half', type=str, help="Cuda half mode, more performance for hardly less quality, False or True", default="True", action="store")
    parser.add_argument('-deflicker', type=str, help="deflicker the depth scan in order to normalize the output, True or False", default="False", action="store")
    args = parser.parse_args()

    main(args.deflicker, args.half, args.model_type, args.height, args.width)