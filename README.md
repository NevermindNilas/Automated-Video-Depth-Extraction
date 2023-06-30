# Automated-Video-Depth-Extraction 
Real-time depth extraction from image and video using with **intel-isl/MiDaS** depth extraction model.

## Dependencies
If you have a cuda capable GPU ( Preferably Nvidia Pascal and onwards )
 - pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Otherwise use this
 - pip3 install torch torchvision torchaudio

Extra requirements
 - pip install -r requirements.txt

## Acknowledgements

 - [isl-org/Midas](https://github.com/isl-org/MiDaS)
 - [egemengulpinar](https://github.com/egemengulpinar/depth-extraction)


## Roadmap
 - FrameSkip - only do the depth scan on every 2nd frame and interpolate using VFI every other frame.
 - FFmpegCV / Decord / PyAV / Vapoursynth for faster decode/encode and hopefully multithreadding
 - Increse CUDA utilization ( current peak is about 70-75% on my 3090 )
 - Is this even fp16???

## Usage/Examples
Place all of your files inside the input folder.
in -output use the output file name only, not entire path, it will be exported to the folder named output within the same directory

currently available commands:
- -height , -width , -output, -model_type ( coming: -codec & probably -out_path )

Example code to run in shell:
- python extraction.py --height 1280 --width 704 --output output.mp4
