# GridTrackNet
A Tensorflow implementation of GridTrackNet for real time tennis ball tracking; a CNN aimed at locating and tracking a small fast moving object throughout multiple concurrent frames by means of grid outputs. 

Official paper: **LINK TO PAPER**

Based on TrackNet: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

## Main changes to TrackNetv2
1. Removed upsampling layers: output consists of three 48x27 grids per frame: confidence grid, x-offset grid, and y-offset grid.
2. 5 input frames and 5 output frames
3. Increased input resolution from 512x288 to 768x432

## GridTrackNet vs TrackNetv2 Comparison:

|Metric|TrackNetv2|**GridTrackNet**|   
|---------|-----|-----|
|Input/output frames|3/3| **5/5**|
|Image resolution|512 x 288| **768 x 432**|
|Inference speed|FPS ON 3080|**FPS ON 3080**|
|Accuracy|0.7501|**NEW VAL**|
|Precision|0.8721|**NEW VAL**|
|Recall|0.8386|**NEW VAL**|
|F1|0.8550|**NEW VAL**|

Accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)

Precision = TP / (TP + FP1 + FP2)

Recall = TP / (TP + FN)

F1 = (2*(Precision*Recall))/(Precision + Recall)

TP (True Positive): when predicted ball location is less than 4 pixels from true ball location.

TN (True Negative): when the model correctly predicts no ball in a frame.

FP1 (False Positive Type 1): when the model correctly predicts the presence of a ball within a frame, but outside the tolerance value of 4 pixels.

FP2 (False Positive Type 2): when the model predicts the presence of a ball within a frame while there is no ball visible. 

FN (False Negative): when the model fails to make a prediction in a frame while it should have.


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



