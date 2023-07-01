


#Automated-Video-Depth-Extraction 
Real-time depth extraction from image and video using with **intel-isl/MiDaS** depth extraction model.

## Dependencies
If you have a cuda capable GPU ( Preferably Nvidia Pascal and onwards )
 - pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

If not
 - pip3 install torch torchvision torchaudio

extra requirements
 - pip install -r requirements.txt



## Acknowledgements

 - [isl-org/Midas](https://github.com/isl-org/MiDaS)
 - [egemengulpinar/depth-extraction](https://github.com/egemengulpinar/depth-extraction)


## Roadmap
 - FrameSkip - only do the depth scan on every 2nd frame and interpolate using VFI every other frame. ( half implemented )
 - Add Multi Threadding ( should be relatively Easy )
 - FFMPEG / Decord / PyAV / Vapoursynth for faster decode/encode ( IO is currently not a bottleneck, with the exception of encoding, might ditch this idea )
 - Increse CUDA utilization ( current peak is about 70-75% on my 3090 ) ! (added half precision, now it's about 40-50% meaning with threadding / multiprocessing I can potentially double the performance)
 - ̶I̶s̶ ̶t̶h̶i̶s̶ ̶e̶v̶e̶n̶ ̶f̶p̶1̶6̶?̶?̶?
 - Add Deflickering in post in order to minimize artifacting ( Not sure just how yet )

## Usage/Examples
Place all of your files inside the input folder


currently available commands:
 - -video -height -width -output -half ( coming: -codec, -skip )

Example code to run in shell:
 - python inference.py -video -height 1280 -width 704 -output output.mp4 -half True


## Demo

![input](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/input/input.gif)![output](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/output/output.gif)

 - Do keep in mind that this is compressed, take that info as you will.


## Update Log:
 
[July 1st 2023]
 
 - Split Extraction.py into inference.py and depth_extract.py for better visibility, it was starting to become a nightmare
 - Added Cuda Half precision ( somehow, still figuring out how all of these numpy arrays work )
 - Updated Readme with more info.
 - Partly added -Skip, it still need Rife
 - Might ditch the idea of using something other than OpenCV for it is not the true bottleneck afaik.
 - Added Deflickering as a Roadmap, the current models have a tendency to 'over'flicker and I think this might help or not.
 - With Lower Cuda utilization, I can now think of adding threading in order to maximixe load across the GPU.