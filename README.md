


# Automated-Video-Depth-Extraction 
Effortlessly achieve real-time depth extraction from videos using the advanced intel-isl/MiDaS depth extraction model, eliminating the need for cumbersome frame extraction.

## Dependencies
Ensure the presence of a CUDA-capable GPU, preferably Nvidia Pascal and onwards, for optimal performance.
 > pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For systems without a CUDA-capable GPU:
 > pip3 install torch torchvision torchaudio

Install additional requirements:
 > pip install -r requirements.txt

## Acknowledgements
Gratitude extended to the following contributors and projects:
 - [isl-org/Midas](https://github.com/isl-org/MiDaS)
 - [egemengulpinar/depth-extraction](https://github.com/egemengulpinar/depth-extraction)

## Roadmap
Enhance functionality through the following roadmap features:

- FrameSkip: Implement depth scan on every 2nd frame and interpolate using VFI every other frame.
- Is this even FP16? (Yes, it is now :D)

## Usage/Examples
Organize your files within the designated input folder. Execute the following command in the terminal:

Currently available commands include:

- -height
- -width
- -half
- -nt

Example code to run in terminal:
 > python inference.py -video -height 1280 -width 720 -half True -nt 2

## Demo

![input](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/input/input.gif)![output](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/output/output.gif)

 - Note: Images are compressed; consider this in your assessment.

Explore the GitHub repository for detailed information and updates. Your feedback and contributions are greatly appreciated!