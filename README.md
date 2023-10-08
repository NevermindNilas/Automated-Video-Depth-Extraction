


# Automated-Video-Depth-Extraction 
Real-time depth extraction from video using with **intel-isl/MiDaS** depth extraction model with absolutely no frame extraction.

## Dependencies
If you have a cuda capable GPU ( Preferably Nvidia Pascal and onwards )
 > pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

If not
 > pip3 install torch torchvision torchaudio

extra requirements
 > pip install -r requirements.txt

## Acknowledgements

 - [isl-org/Midas](https://github.com/isl-org/MiDaS)
 - [egemengulpinar/depth-extraction](https://github.com/egemengulpinar/depth-extraction)

## Roadmap
 - FrameSkip - only do the depth scan on every 2nd frame and interpolate using VFI every other frame.
 - ̶I̶s̶ ̶t̶h̶i̶s̶ ̶e̶v̶e̶n̶ ̶f̶p̶1̶6̶?̶?̶? ( it is now :D )

## Usage/Examples
Place all of your files inside the input folder.

currently available commands:
 - -height -width -half -deflicker ( coming: -codec, -skip )

Example code to run in terminal:
 - python inference.py -video -height 1280 -width 704 -half True

## Demo

![input](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/input/input.gif)![output](https://github.com/NevermindNilas/Automated-Video-Depth-Extraction/blob/main/output/output.gif)

 - Do keep in mind that this is compressed, take that info as you will.