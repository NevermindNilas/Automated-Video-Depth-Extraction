import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import argparse
import time
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
if device.type == 'cuda':
    model = model.to(device)
model.eval()

# Transform the input
transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),  # Resize to dimensions divisible by 32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def depth_extract_image(path):

    depth_map = depth_extract(path)
    # Display the depth map
    cv2.imshow('Depth Map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def depth_extract(frame):
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
    return depth_map


def depth_extract_video(path):
    video = cv2.VideoCapture(path)

    # Check if the video is opened successfully.
    if not video.isOpened():
        print("Error opening video file")
        return
    
    while True:
        # Read a frame from the video.
        ret, frame = video.read()
        # Check if the frame is read successfully.
        if not ret:
            break
        cv2.imshow('Frame', depth_extract(frame))
 

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    # Release the video file and close any open windows.
    video.release()
    cv2.destroyAllWindows()


def main(video,image,path):
    if image:
        depth_extract_image(path)
    elif video:
        depth_extract_video(path)





if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")

    parser.add_argument(
        "--path", type=str, default=None, help="Directory path containing images"
    )
    parser.add_argument(
        "--image",
          help="image depth extraction",
          default=None,
          action="store_true")
    parser.add_argument(
        "--video",
          help="video frame depth extraction",
          default=None,
          action="store_true")
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.video, args.image, args.path)
