# GridTrackNet
A Tensorflow implementation of GridTrackNet for real time tennis ball tracking; a CNN aimed at locating and tracking a small fast moving object throughout multiple concurrent frames by means of grid outputs.

<p align="center">
  <img src="https://github.com/VKorpelshoek/GridTrackNet/blob/main/Figures/GridTrackNet.png" alt="image" style="display:block; margin:auto;" />
</p>


Official paper: **LINK TO PAPER**

Based on TrackNetv2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

Disclaimer: some parts of the source code have been developed in assistance with ChatGPT and, even though unlikely, might contain unexpected behavior at times.

<p align="center">
  <img src="https://github.com/VKorpelshoek/GridTrackNet/blob/main/Figures/GridTrackNet%20Preview%20GIF.gif" alt="image" style="display:block; margin:auto;" />
</p>

## Main changes to TrackNetv2
1. Removed upsampling layers for faster inference: output consists of three 48x27 grids per frame: confidence grid, x-offset grid, and y-offset grid.
2. 5 input frames and 5 output frames
3. Increased input resolution from 512x288 to 768x432

### GridTrackNet vs TrackNetv2 Comparison:

|Metric|TrackNetv2|**GridTrackNet**|   
|---------|-----|-----|
|Input/output frames|3/3| **5/5**|
|Image resolution|512 x 288| **768 x 432**|
|Inference speed|FPS ON 3080|**FPS ON 3080**|
|Accuracy|0.7501|**NEW VAL**|
|Precision|0.8721|**NEW VAL**|
|Recall|0.8386|**NEW VAL**|
|F1|0.8550|**NEW VAL**|

Note: metrics were computed only once on a separate test dataset.

### Formulas
- Accuracy = \(\frac{{TP + TN}}{{TP + TN + FP1 + FP2 + FN}}\)
- Precision = \(\frac{{TP}}{{TP + FP1 + FP2}}\)
- Recall = \(\frac{{TP}}{{TP + FN}}\)
- F1 = \(\frac{{2 \cdot (Precision \cdot Recall)}}{{Precision + Recall}}\)

### Formula Variable Definitions
- TP (True Positive): when the model correctly predicts the location of a ball within a frame being less than 4 pixels from the true ball location.
- TN (True Negative): when the model correctly predicts no ball visible within a frame.
- FP1 (False Positive Type 1): when the model correctly predicts the presence of a ball within a frame, but outside the tolerance value of 4 pixels from the true ball location.
- FP2 (False Positive Type 2): when the model incorrectly predicts the presence of a ball within a frame while there is no ball visible. 
- FN (False Negative): when the model incorrectly predicts the absence of a ball within a frame while there is a ball visible. 




## Setup
HERE IS HOW TO SETUP requirements.txt
Also installation of cuda

## Inference API / Video
ADD HERE THE API FOR THE INFERENCE OR VIDEO GENERATION

## Custom Training Guide
1. Per video, use FramGenerator.py to extract individual frames from a video.
2. Per match folder, use LabellingTool.py to label all frames.
3. After all match folders are annotated, use the DataGen.py to generate the dataset in TFRecord format.
4. Train
5. Inference

Resulting Sample Dataset Folder Structure:
```
Dataset
|   
|___match1    
|       |    
|       |___ frames
|       |     |___0.png
|       |     |___1.png
|       |     |...
|       |     |___x.png
|       |
|       |____ Labels.csv
|     
|___match2    
|       |    
|       |___ frames
|       |     |___0.png
|       |     |___1.png
|       |     |...
|       |     |___x.png
|       |
|       |____ Labels.csv
| 
|...
|
|___matchX    
|       |    
|       |___ frames
|       |     |___0.png
|       |     |___1.png
|       |     |...
|       |     |___x.png
|       |
|       |____ Labels.csv

|___TFRecordFiles    
|       |___train0.tfrecord
|       |___train1.tfrecord
|       ...
|       |___trainX.tfrecord
|       |___val.tfrecord
```  

### Frame Generator
Outputs individual frames with 1280x720 resolution from an input video.

Note: input video format must .mp4, be either 30FPS or 60FPS, and at least 1280x720 resolution. The export directory should end with 'matchX', where X is an index (first index is 1.) 

Example usage:
```bash
python "/path/to/FrameGenerator.py" --video_dir="path/to/video.mp4" --export_dir="path/to/Dataset/matchX"
```   

### Labelling Tool
Outputs a Labels.csv file in the /matchX/frames folder containing the pixel coordinate and visibility per frame.

Note: you can only save the annotations when all frames have been annotated with either a coordinate of the ball, or with the 'invisible' state. It is advised to use a mouse with a scroll wheel for zooming capabilities. For faster annotation speeds, the next frame is automatically loaded after annotating the previous frame.

Example usage:
```bash
python "/path/to/LabellingTool.py" --frames_dir="path/to/Dataset/matchX/frames/"
```   

Controls:
|Type|Event|Function|   
|-----|----|--------|   
|Mouse|Left mouse click|Mark ball location|
|Mouse|Scroll wheel|Zoom in/zoom out|
|Key|a|Previous frame|
|Key|d|Next frame|
|Button|Toggle State|Specify the presence of a ball in a frame|
|Button|Remove Pixel|Removes current ball annotation from the frame|
|Button|Remove Frame|Removes current frame from the 'frames folder'|
|Button|Save Results|Saves all annotations to \matchX\frames\Labels.csv|




### Dataset Generation
The dataset consists of x images from 81 different tennis video's, combined into x training instances and x validation instances. 

Link: ....

Example usage:
```bash
python "/path/to/DataGen.py" --input_dir="path/to/your/matches/folder" --export_dir="path/to/your/export/folder" --val_split=0.2 --augment_data=1 --next_img_index=2
```
Accepted arguments:
```bash

  -h, --help            show this help message and exit
  --input_dir (required)
        Input directory of the folder containing all folders
        with names with the prefix 'match'.
  --export_dir (required)
        Export directory where the data will be saved.
  --augment_data {0,1} (optional) 
        Boolean indicating whether or not the data should be
        augmented as well (flipped horizontally). 1 for
        augmentations, 0 for no augmentations. No augmented
        versions will be used as validation instances. Default
        = 1
  --val_split (optional)
        Fraction of instances to be used for validation: must
        be greater than 0.0 and less than 1.0. Note this only
        affects the validation:non-augmented-data ratio, not
        the total validation:train-instances ratio. Default =
        0.2
  --next_img_index (optional)
        Specifies the overlap of images between instances;
        specifically the integer to be used for selecting the
        index of the first image of the next instance relative
        to the index of the first image of the previous
        instance. For example, if set to 2, the first instance
        will contain images with indices [0,1,2,3,4] and the
        second instance will contain images with indices
        [2,3,4,5,6]. Default = 2
```


### Training


## Architecture
Adapted version of the VGG16 model. 
|Layer Number|Layer Type|Filters|Kernel Size|Activation|Output Resolution|   
|-|-----|-------|-----------|----------|---|
|1|Conv2D|64|3x3|ReLU + Batch Norm.|768 x 432|
|2|Conv2D|64|3x3|ReLU + Batch Norm.|768 x 432|
|3|MaxPool2D|-|2x2 pooling, 2x2 strides|-|384 x 216|
|4|Conv2D|128|3x3|ReLU + Batch Norm.|384 x 216|
|5|Conv2D|128|3x3|ReLU + Batch Norm.|384 x 216|
|6|MaxPool2D|-|2x2 pooling, 2x2 strides|-|192 x 108|
|7|Conv2D|256|3x3|ReLU + Batch Norm.|192 x 108|
|8|Conv2D|256|3x3|ReLU + Batch Norm.|192 x 108|
|9|MaxPool2D|-|2x2 pooling, 2x2 strides|-|96 x 54|
|10|Conv2D|256|3x3|ReLU + Batch Norm.|96 x 54|
|11|Conv2D|256|3x3|ReLU + Batch Norm.|96 x 54|
|12|Conv2D|256|3x3|ReLU + Batch Norm.|96 x 54|
|13|MaxPool2D|-|2x2 pooling, 2x2 strides|-|48 x 27|
|14|Conv2D|512|3x3|ReLU + Batch Norm.|48 x 27|
|15|Conv2D|512|3x3|ReLU + Batch Norm.|48 x 27|
|16|Conv2D|512|3x3|ReLU + Batch Norm.|48 x 27|
|**17**|**Conv2D**|**15**|**3x3**|**Sigmoid**|**48 x 27**|









