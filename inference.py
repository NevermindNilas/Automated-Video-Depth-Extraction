import torch
import torchvision.transforms as transforms
import argparse
import os
import sys
from depth_extract import depth_extract_video
from deflicker import depth_extract_deflicker

def load_device(half, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if half == "True":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).to(device)
    model.eval()

    return device, model

def main(deflicker, half, model_type, height, width, nt):
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    width = (width // 32 + 1) * 32
    height = (height // 32 + 1) * 32

    device, model = load_device(half, model_type)

    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()

    if not video_files:
        sys.exit("No videos found in the input folder")

    for i, video_file in enumerate(video_files):
        output = os.path.splitext(video_file)[0] + ".mp4"
        output_path = os.path.join(output_path, output)
        video_file = os.path.join(input_path, video_file)
        print("Processing Video File:", os.path.basename(video_file))
        if deflicker == "True":
            depth_extract_deflicker(video_file, output_path, width, height, model, device, half)
        else:
            depth_extract_video(video_file, output_path, width, height, model, device, half, nt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument('-width', type=int, help="Width of the corresponding output, must be a multiple of 32", default=None)
    parser.add_argument("-height", type=int, help="Height of the corresponding output, must be a multiple of 32", default=None)
    parser.add_argument('-model_type', required=False, type=str, help="Which MIDAS model to choose from, e.g DPT_Large, DPT_Hybrid or path/to/model, custom isn't functional for now.", default="DPT_Hybrid", action="store")
    parser.add_argument('-half', type=str, help="Cuda half mode, more performance for hardly less quality, False or True", default="True", action="store")
    parser.add_argument('-deflicker', type=str, help="deflicker the depth scan in order to normalize the output, True or False", default="False", action="store")
    parser.add_argument('-nt', type=int, help="Number of threads to use, default is 1", default=1, action="store")
    args = parser.parse_args()
    main(args.deflicker, args.half, args.model_type, args.height, args.width, args.nt)