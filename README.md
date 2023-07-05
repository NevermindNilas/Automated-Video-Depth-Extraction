


# Automated-Video-Depth-Extraction 
Real-time depth extraction from video using with **intel-isl/MiDaS** depth extraction model with absolutely no frame extraction.

## Dependencies
Python 3.9 or 3.11 ( I have only tested these )

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
 - Add Multi Threadding ( should be relatively Easy ) ( update, it isn't for some reason ) ( added but it made things worse )
 - FFMPEG / Decord / PyAV / Vapoursynth for faster decode/encode ( IO is currently not a bottleneck, with the exception of encoding, might ditch this idea ) ( no point anymore )
 - Increse CUDA utilization ( current peak is about 70-75% on my 3090 ) ! (added half precision, now it's about 40-50% meaning with threadding / multiprocessing I can potentially double the performance) ( increased it but it was not worth it )
 - ̶I̶s̶ ̶t̶h̶i̶s̶ ̶e̶v̶e̶n̶ ̶f̶p̶1̶6̶?̶?̶? ( it is now :D )
 - Add Deflickering in post in order to minimize artifacting ( added a very basic deflickering technique, play around with it )

## Usage/Examples
Place all of your files inside the input folder.

currently available commands:
 - -video -height -width -output -half -deflicker -nt ( coming: -codec, -skip )

Example code to run in shell:
 - python inference.py -video -height 1280 -width 704 -half True

## Demo

![input](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/input/input.gif)![output](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/output/output.gif)

 - Do keep in mind that this is compressed, take that info as you will.

## Update Log:
[ July 5th 2023 ]

 - Managed to break the encoder somehow, idk how just yet but I will look into it :D
 - Split video processing into buffer and processing, there's now a thread for video buffering and another set for video depth scanning that's determined by -nt, added an extra 10% in performance ( from 9.1 to 10.2 in personal testing )
 - Added MultiThreadding and -nt, for 1280x704 content, -half True and 3090 it can be used with 2 threads. MT seems to actually worsen the performance in my testing but I need to further look into it, I may have f up something.
 - the output container no longer has to be declared, so you can just use output and it will default to .mp4 and if the whole -output ouptput.mp4 is not declared in the CLI, it will just use input video file name + .mp4
 - improved the code a bit, -skip is around the corner as soon as I realise how to add Rife, I'd like to use ncnn vulkan but there's only .png/.jpg approaches and I'd like to only use numpy arrays with no frame extraction / frame save, preferably.

[ July  2nd 2023 ]

 - Implemented -deflicker and deflickering, overall it's pretty meh, it only hurts the performance by a negligible margin ( 0.5% to 1% ), I will improve it further more down the line.
 - Added cupy for an extra 1 FPS :D
 - Cleaned up the code a bit, I was also thinking to use pep8
 - Added more accurate FPS logging and also an Average FPS.
 - MultiThreadding proved to be a bit more challenging than I've thought initially.
 
[ July 1st 2023 ]
 
 - Split Extraction.py into inference.py and depth_extract.py for better visibility, it was starting to become a nightmare
 - Added Cuda Half precision ( barely, still figuring out how all of these numpy arrays work )
 - Updated Readme with more info.
 - Partly added -Skip, it still need Rife
 - Might ditch the idea of using something other than OpenCV for it is not the true bottleneck as far as I can tell.
 - Added Deflickering into the Roadmap, the current models have a tendency to 'over'flicker and I think this might help, maybe.
 - With Lower Cuda utilization, I can now think of adding threading in order to maximixe load across the GPU.