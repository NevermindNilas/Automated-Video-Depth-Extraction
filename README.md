# depth-extraction
Real-time depth extraction from image and video using with **intel-isl/MiDaS** depth extraction model.
If you have CUDA support, I strongly recommend to use with GPU rather CPU.

Please refer to transformers.resize() for speed up the project.
## Demo


https://github.com/egemengulpinar/depth-extraction/assets/71253469/9eaa0d18-d9d0-45c1-8ea8-78dfd9d52917





## Usage

```shell
python midas_depth_extraction.py --video --path csgo_walk_2.mp4
```

## Arguments 
| Name             | Type | Description 
| ----------------- | ------------- | ----------- |
| --video | **bool** | video frame depth extraction |
| --image | **bool** | image depth extraction |
| --path | **str** | Absolute path to the image/video |
