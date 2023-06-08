# GridTrackNet

<p align="center">
  <img src="https://github.com/VKorpelshoek/GridTrackNet/blob/main/Figures/GridTrackNet.png" alt="image" style="display:block; margin:auto;" />
</p>

Official Tensorflow implementation of GridTrackNet for real time tennis ball tracking; a CNN aimed at locating and tracking a small fast moving object throughout multiple concurrent frames by means of grid outputs.

Paper: *Coming soon...*

Originally based on TrackNetV2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2 <sup>1</sup>

### Main improvements
1. 🚀 Redesigned ultra-efficient architecture reaching 115 FPS (📈+238% to TrackNetV2)*
2. 🎾 Increased input resolution from 512x288 to 768x432
3. 📺 5 input frames and 5 output frames per instance for increased temporal context

**Benchmarks performed on 10-core M1 Pro MacBook Pro 2021 with Tensorflow Metal Version*

<p align="center">
  <img src="https://github.com/VKorpelshoek/GridTrackNet/blob/main/Figures/GridTrackNet%20Preview%20GIF.gif" alt="image" style="display:block; margin:auto;" />
</p>

### GridTrackNet vs TrackNetV2 Comparison:

||TrackNetV2|**GridTrackNet**|   
|---------|-----|-----|
|Input/output frames|3/3| **5/5**|
|Input resolution|512 x 288| **768 x 432**|
|Output resolution|512 x 288| **48 x 27**|
|Inference speed|34 FPS|**115 FPS**|
|Accuracy|89.0%|**90.8%**|
|Precision|94.1%|**95.2%**|
|Recall|93.7%|**94.7%**|
|F1|93.9%|**94.9%**|

**Note: metrics were computed only once on a separate test dataset of sufficient size.*

## Setup
1. Follow the complete Tensorflow installation guide for the installation on your system and how to enable hardware acceleration.
      - Linux/Windows: https://www.tensorflow.org/install/pip
      - Mac: https://developer.apple.com/metal/tensorflow-plugin/

2. Create a virtual environment (Conda, Miniforge, etc.) using `python=3.10.8`
3. In your virtual environment, run:
```commandline
pip install -r "path/to/requirements.txt"
```

## Inference
### Video Output
`Predict.py` receives as input a `.mp4` video and outputs the same video with visual predicted ball locations.

Example usage:
```commandline
python "/path/to/Predict.py" --video_dir="/path/to/video.mp4" --model_dir="/path/to/model_weights.h5" --display_trail=1
``` 



|Argument|Description|  
|-----|----|
|video_dir (required) | Path to a `.mp4` video|
|model_dir (optional) | Path to `model_weights.h5` file for loading a custom model|
|display_trail (optional) | Displays a yellow trail of the ball trajectory. If set to 0, only a red circle around the predicted ball location is displayed on each frame.|

### API
`Predict.py` script can be imported in your own code and be called by the following function:
```commandline 
Predict.getPredictions(frames, isBGRFormat = False)
```

Receives as input a list of concurrent frames (number of frames should be a multiple of 5), and outputs a list of pixel coordinates for each input frame. If no ball was detected, the model returns coordinate (0,0). In case the frames are in BGR format (such as when using OpenCV), specify this with the `isBGRFormat` argument.

## Custom Data Training Guide
### Overview
0. Trim your custom videos to contain only a single rally with your own video trimming software.
1. For each trimmed video, use `FrameGenerator.py` to extract the individual frames of the video.
2. For each match folder, use `LabellingTool.py` to label all frames.
3. After annotating all data, use `DataGen.py` to generate the dataset in TFRecord format.
4. Train the model using `Train.py`.
5. Deploy your custom model! You can use `Predict.py` by specifying the path to the saved `.h5` file with the argument `--model_dir`

*More detailed explanations for each utility can be found below. For each utility, the `-h` flag can be used to check supported arguments.*

Resulting sample dataset folder structure:
```
Dataset
|   
|___match1    
|       |___ frames
|       |     |___0.png
|       |     |___1.png
|       |     |...
|       |     |___x.png
|       |
|       |____ Labels.csv   
|
|___match2     
|       |___ frames
|       |     |___0.png
|       |     |___1.png
|       |     |...
|       |     |___x.png
|       |
|       |____ Labels.csv
|...
|
|___matchX     
|       |___ frames
|       |     |___0.png
|       |     |___1.png
|       |     |...
|       |     |___x.png
|       |
|       |____ Labels.csv
|
|___TFRecordFiles    
|       |___train0.tfrecord
|       |___train1.tfrecord
|       ...
|       |___trainX.tfrecord
|       |___val.tfrecord
```  

### 1. Frame Generator
`FrameGenerator.py` outputs individual frames with 1280x720 resolution from an input video.

Note: input video format must be `.mp4`, be either 30FPS or 60FPS, and at least 1280x720 resolution. The export directory should end with `matchX`, where `X` is an index (first index is 1.)  See the example folder structure above.

Example usage:
```commandline
python "/path/to/FrameGenerator.py" --video_dir="path/to/video.mp4" --export_dir="path/to/Dataset/matchX"
```   

### 2. Labelling Tool
`LabellingTool.py` outputs a `Labels.csv` file containing the pixel coordinates of the ball and visibility per frame.

Label the frame by clicking on the center of a ball. In case of elongated, blurred, or almost invisible balls, try to still annotate the center. Specify `VISIBLE` for when a ball is (partially) visible in a frame, and `INVISIBLE` when it is occluded or out of frame. 

It is advised to use a mouse with a scroll wheel for zooming capabilities. When using the scroll wheel, the frame will be zoomed in at the place below the mouse pointer. Note that, for faster annotation speeds, the next frame is automatically loaded after annotating the previous frame with the same zoom level as previous annotation.

You can save the annotations only after all frames have been annotated, either with a ball coordinate or with the 'invisible' state. If the `Save Results` button is pressed without annotating every frame, the indices of the frames with missing labels are printed to the console. The program automatically terminates if the `Labels.csv` file is successfully saved. **Important: if the program is terminated BEFORE saving, no labels are saved!**

Example usage:
```commandline
python "/path/to/LabellingTool.py" --match_dir="path/to/Dataset/matchX"
```   

Controls:
|Type|Event|Function|   
|-----|----|--------|   
|Mouse|Left mouse click|Mark ball location|
|Mouse|Scroll wheel|Zoom in/zoom out at place of mouse pointer|
|Key|a|Previous frame|
|Key|d|Next frame|
|Button|Toggle State|Specify the presence of a ball in a frame|
|Button|Remove Pixel|Removes current ball annotation from the frame|
|Button|Remove Frame|Permanently deletes the current frame from the frames directory|
|Button|Save Results|Saves all annotations to `\matchX\Labels.csv`|




### 3. Dataset Generation
`DataGen.py` generates TFRecord files containing the instances with corresponding labels to be used for training.

Link to the original dataset (48.2GB): https://drive.google.com/drive/folders/1FzkE5i5_ybyn6Tc6KMj0mgTiH7zPGgHm?usp=sharing

*Training data consisted primarily of diverse amateur footage as well as professional TV broadcasts, but can be trained on custom data for your own use case.*

Example usage:
```commandline
python "/path/to/DataGen.py" --input_dir="path/to/your/matches/folder" --export_dir="path/to/your/export/folder" --val_split=0.2 --augment_data=1 --next_img_index=2
```
Accepted arguments:
|Argument|Description|  
|-----|----|
|input_dir (required)|Input directory of the folder containing all folders with names with the prefix `match`.
|export_dir (required)| Export directory where the data will be saved.
|augment_data {0,1} (optional) | Boolean indicating whether or not the data should be augmented as well (flipped horizontally). 1 for augmentations, 0 for no augmentations. No augmented instances will be used as validation instances. Default = 1
|val_split (optional)|Fraction of instances to be used for validation: must be greater than 0.0 and less than 1.0. Note this only affects the validation:non-augmented-data ratio, not the total validation:train-instances ratio. Default = 0.2
|next_img_index (optional)|Specifies the overlap of images between instances; specifically the integer to be used for selecting the index of the first image of the next instance relative to the index of the first image of the previous instance. For example, if set to 2, the first instance will contain images with indices [0,1,2,3,4] and the second instance will contain images with indices [2,3,4,5,6]. For no overlap, set to 5. Default = 2



### 4. Training
`Train.py` can be used to train the GridTrackNet model with custom data.

```commandline
python "/path/to/Train.py" --data_dir="path/to/tfrecord/files" --save_weights="path/to/your/export/folder" --epochs=50 --tol=4
```
Accepted arguments:
|Argument|Description|  
|-----|----|
|data_dir (required)|Data directory of the folder containing all folders with names with the prefix `match`.
|load_weights (optional)|Directory to load pre-trained weights.
|save_weights (required)|Directory to store model weights and training metrics.
|epochs (required)|Number of epochs (iterations of the training data) the model should be trained for.|
|tol (optional)|Specifies the tolerance of the model: the number of pixels the predicted location is allowed to deviate from the true location. Default = 4
|batch_size (optional) | Specify the batch size to train on. Default = 3|

## Architecture
Adapted version of the VGG16 model.<sup>2</sup> 
|Layer Number|Layer Type|Filters|Kernel Size|Activation|Output Resolution|   
|-|-----|-------|-----------|----------|---|
|1|Conv2D|64|3x3|ReLU + BN|768 x 432|
|2|Conv2D|64|3x3|ReLU + BN|768 x 432|
|3|MaxPool2D|-|2x2 pooling|-|384 x 216|
|4|Conv2D|128|3x3|ReLU + BN|384 x 216|
|5|Conv2D|128|3x3|ReLU + BN|384 x 216|
|6|MaxPool2D|-|2x2 pooling|-|192 x 108|
|7|Conv2D|256|3x3|ReLU + BN|192 x 108|
|8|Conv2D|256|3x3|ReLU + BN|192 x 108|
|9|MaxPool2D|-|2x2 pooling|-|96 x 54|
|10|Conv2D|256|3x3|ReLU + BN|96 x 54|
|11|Conv2D|256|3x3|ReLU + BN|96 x 54|
|12|Conv2D|256|3x3|ReLU + BN|96 x 54|
|13|MaxPool2D|-|2x2 pooling|-|48 x 27|
|14|Conv2D|512|3x3|ReLU + BN|48 x 27|
|15|Conv2D|512|3x3|ReLU + BN|48 x 27|
|16|Conv2D|512|3x3|ReLU + BN|48 x 27|
|**17**|**Conv2D**|**15**|**3x3**|**Sigmoid**|**48 x 27**|

*BN = Batch Normalization
## Formulas
- Accuracy = $\dfrac{TP + TN}{TP + TN + FP1 + FP2 + FN}$
- Precision = $\dfrac{TP}{TP + FP1 + FP2}$
- Recall = $\dfrac{TP}{TP + FN}$
- F1 = $\dfrac{2*(Precision * Recall)}{Precision + Recall}$


### Formula Variable Definitions
- TP (True Positive): The model correctly predicts the location of a ball within a frame being less than 4 pixels from the true ball location.
- TN (True Negative): The model correctly predicts no ball visible within a frame.
- FP1 (False Positive Type 1): The model predicts the presence of a ball within a frame, but its predicted location is more than 4 pixels away from the true ball location.
- FP2 (False Positive Type 2): The model incorrectly predicts the presence of a ball within a frame while there is no ball visible. 
- FN (False Negative): The model incorrectly predicts the absence of a ball within a frame while there is a ball visible. 

## References
1. N. -E. Sun et al., "TrackNetV2: Efficient Shuttlecock Tracking Network," 2020 International Conference on Pervasive Artificial Intelligence (ICPAI), Taipei, Taiwan, 2020, pp. 86-91, doi: 10.1109/ICPAI51961.2020.00023.
2. Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. CoRR, abs/1409.1556. 
3. amateur tennis data courtesy to AmateurTennisTV http://amateurtennis.tv. YouTube: https://www.youtube.com/@AmateurTennistv

*Disclaimer: some parts of the source code have been developed in assistance with ChatGPT-4 and, even though unlikely, might contain unexpected behavior at times.*





