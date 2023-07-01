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
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    
    entrypoints = torch.hub.list("intel-isl/MiDaS", force_reload=True)
    print(entrypoints)
    
    # for now I am not sure if it works but my CUDA util went from 75% to 50% and VRAM also dropped heavily so I assume yes
    if half == "True":
        print('Using half precision')
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    if device.type == 'cuda':
        model = model.to(device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    model.eval()
    
    # Transform the input
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),  # Resize to dimensions divisible by 32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return device, model, transform

def main(half, skip, output, model_type, height, width):
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get the torch properties. Maybe declaring it globally was better in terms of startup times
    device, model, transform = load_device(half, model_type, width, height)

    video_files = [
        f for f in os.listdir(input_path) if f.endswith(
            ('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()

    if video_files == 0:
        sys.exit("No videos found in the input folder")

    for video_file in video_files:
        # Get proper paths
        video_file = os.path.join(input_path, video_file)
        output_path = os.path.join(output_path, output)
        print("Processing Video File:", video_file)
        depth_extract_video(video_file, output_path, width, height, model, transform, device, skip, half)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")

    parser.add_argument(
        '-width',
        type=int,
        help="Width of the corresponding output, must be a multiple of 32",
        default=None,
    )

    parser.add_argument(
        "-height",
        type=int,
        help="Height of the corresponding output, must be a multiple of 32",
        default=None,
    )

    parser.add_argument(
        '-model_type',
        required=False,
        type=str,
        help="Which MIDAS model to choose from, e.g DPT_Large, DPT_Hybrid or path/to/model, custom isn't functional for now.",
        default="DPT_Hybrid",
        action="store"
    )

    parser.add_argument(
        '-output',
        type=str,
        help="The output's name",
        default="",
    )

    parser.add_argument(
        '-skip',
        type=str,
        help="Whether to skip every other frame, False or True",
        default="False",
        action="store",
    )
    
    # For some apparent reason, if I used bool ) here, it would not want to take the value False or True, it only took True \n"
    # So I went for string, it should do the job
    parser.add_argument(
        '-half',
        type=str,
        help="Cuda half mode, more performance for hardly less quality, False or True",
        default="True",
        action="store",
    )
    
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.half,args.skip, args.output, args.model_type, args.height, args.width)