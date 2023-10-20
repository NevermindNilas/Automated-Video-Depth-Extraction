import torch
import torchvision.transforms as transforms
import argparse
import os
import sys
from depth_extract import depth_extract_video
from deflicker import depth_extract_deflicker

import warnings # Getting rid of that annoying warning
warnings.filterwarnings("ignore", message="Mapping deprecated model name vit_base_resnet50_384 to current vit_base_r50_s16_384.orig_in21k_ft_in1k.")

def load_device(model_type):
    '''
    I've sort of figured out why other version of Midas wouldn't work,
    it seems like the only compatible version of the library timm that works with
    Midas models 3.1 is timm 0.6.7.
    
    Sadly, whilst I would love to downgrade from the latest available version of timm,
    the performance degradation is too much to ignore, so I'll have to stick with the
    current version.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).to(device)
    model.eval()
    
    return device, model

def main(deflicker, half, model_type, height, width, nt, verbose):
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if nt >= 2:
        print( "====> Number of Extraction threads was set to --", nt, "-- any value equal or greater than 2 might saturate the cuda core count")
    
    if width is None or height is None:
        sys.exit("====> You must specify both width and height <====")
    elif width % 32 != 0:
        print("====> The width is not divisible by 32, rounding up to the nearest multiple of 32 <====")
        width = (width // 32 + 1) * 32
    elif height % 32 != 0:
        print("====> The height is not divisible by 32, rounding up to the nearest multiple of 32 <====")
        height = (height // 32 + 1) * 32
        
    device, model = load_device(model_type)

    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()

    if not video_files:
        sys.exit("No videos found in the input folder")

    for i, video_file in enumerate(video_files):
        output = os.path.splitext(video_file)[0] + ".mp4"
        output_path = os.path.join(output_path, output)
        video_file = os.path.join(input_path, video_file)
        print("\n") # Force new line for each video to make it more user readable
        print("====> Processing Video File:", os.path.basename(video_file), "<====")
        if deflicker == "True":
            depth_extract_deflicker(video_file, output_path, width, height, model, device, half, verbose)
        else:
            depth_extract_video(video_file, output_path, width, height, model, device, half, nt, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument('-width', type=int, help="Width of the corresponding output, must be a multiple of 32", default=None)
    parser.add_argument("-height", type=int, help="Height of the corresponding output, must be a multiple of 32", default=None)
    parser.add_argument('-model_type', required=False, type=str, help="Which MIDAS model to choose from, e.g DPT_Large, DPT_Hybrid or MiDas_small.", default="DPT_Hybrid", action="store")
    parser.add_argument('-half', type=str, help="Cuda half mode, more performance for hardly less quality, False or True, True by default", default="True", action="store")
    parser.add_argument('-deflicker', type=str, help="deflicker the depth scan in order to normalize the output, True or False", default="False", action="store")
    parser.add_argument('-nt', type=int, help="Number of threads to use, default is 1", default=1, action="store")
    parser.add_argument('-v', type=str, help="Show images of the process", default="True", action="store")
    args = parser.parse_args()
    main(args.deflicker, args.half, args.model_type, args.height, args.width, args.nt, args.v)