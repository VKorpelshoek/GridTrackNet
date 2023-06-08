import numpy as np
import csv
from threading import Thread
import os   
from tensorflow.keras.utils import img_to_array, load_img
import cv2
import random
import math
import tensorflow as tf
import argparse

DATA_WIDTH = 1280
DATA_HEIGHT = 720
WIDTH = 768
HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27
IMGS_PER_INSTANCE = 5
GRID_SIZE_COL = DATA_WIDTH / GRID_COLS
GRID_SIZE_ROW = DATA_HEIGHT / GRID_ROWS
BATCH_SIZE = 50

currBatchIdx = batchCount = numTestInstances = numTrainInstances = 0

parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

parser.add_argument('--input_dir', required=True, type=str, help="Input directory of the folder containing all folders with names with the prefix 'match'.")
parser.add_argument('--export_dir', required=True, type=str, help="Export directory where the data will be saved.")
parser.add_argument('--augment_data', required=False, type=int,  choices=range(0, 2), default=1, help='Boolean indicating whether or not the data should be augmented as well (flipped horizontally). 1 for augmentations, 0 for no augmentations. No augmented instances will be used as validation instances. Default = 1')
parser.add_argument('--val_split', required=False, type=float, default=0.2, help='Fraction of instances to be used for validation: must be greater than 0.0 and less than 1.0. Note this only affects the validation:non-augmented-data ratio, not the total validation:train-instances ratio. Default = 0.2')
parser.add_argument('--next_img_index', required=False, type=int, default=2, choices=range(1,IMGS_PER_INSTANCE+1), help='Specifies the overlap of images between instances; specifically the integer to be used for selecting the index of the first image of the next instance relative to the index of the first image of the previous instance. For example, if set to 2, the first instance will contain images with indices [0,1,2,3,4] and the second instance will contain images with indices [2,3,4,5,6].Default = 2')

args = parser.parse_args()

INPUT_DIR = args.input_dir
EXPORT_DIR = args.export_dir
AUGMENT_DATA = bool(args.augment_data)
VAL_SPLIT = args.val_split
NEXT_IMG_INDEX = args.next_img_index

if(VAL_SPLIT <= 0.0 or VAL_SPLIT >= 1.0):
    print("\nPARSE ERROR: Input argument 'val_split' must be greater than 0.0 and less than 1.0.")
    exit(1)

#Helper function to convert data into raw bytes
def bytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#Stores image and label instances together by serialization
def serializeExample(image, label):
    feature = {
        'image': bytesFeature(image.tobytes()),
        'label': bytesFeature(label.tobytes()),
    }
    exampleProto = tf.train.Example(features=tf.train.Features(feature=feature))
    return exampleProto.SerializeToString()

#Turns images with target's coordinates into data and label instances
#Also augmentates data by horizontal flipping if specified
def getDataAndLabels(imgs,xCoords,yCoords,visibilities):
    global IMGS_PER_INSTANCE
    global GRID_SIZE_COL
    global GRID_SIZE_ROW

    dataEntry = []
    labelEntry = []
    for k in range(0, IMGS_PER_INSTANCE):
        img = imgs[k]
        currX = xCoords[k]
        currY = yCoords[k]

        confGrid = np.zeros((GRID_ROWS, GRID_COLS),dtype=np.float32)
        xOffsetGrid = np.zeros((GRID_ROWS, GRID_COLS),dtype=np.float32)
        yOffsetGrid = np.zeros((GRID_ROWS, GRID_COLS),dtype=np.float32)

        xPos = currX / GRID_SIZE_COL
        yPos = currY / GRID_SIZE_ROW

        xCellIndex = math.floor(xPos)
        yCellIndex = math.floor(yPos)
        xOffset = xPos - xCellIndex
        yOffset = yPos - yCellIndex
    
        #if(visibilities[k] == 1):
        if(not ((visibilities[k] == 0) and xCoords[k] == 0 and yCoords[k] == 0)):
            confGrid[yCellIndex, xCellIndex] = 1
            xOffsetGrid[yCellIndex, xCellIndex] = xOffset
            yOffsetGrid[yCellIndex, xCellIndex] = yOffset
        
        labelEntry.append(np.stack((confGrid, xOffsetGrid, yOffsetGrid),axis=-1))

        img = cv2.resize(img, (WIDTH,HEIGHT))
        img = img.astype(np.float32)
        img = img / 255
        img = np.moveaxis(img, -1, 0)
        dataEntry.append(img[0])
        dataEntry.append(img[1])
        dataEntry.append(img[2])

    dataEntry = np.asarray(dataEntry).astype(np.float32)  
    labelEntry = np.asarray(labelEntry).astype(np.float32)

    if(AUGMENT_DATA):
        dataEntryAugmented = np.flip(dataEntry,axis=2)    
        confGrid, xOffsetGrid, yOffsetGrid = np.split(labelEntry, 3, axis=-1)
        confGrid = np.squeeze(confGrid, axis=-1)
        confGridFlipped = np.flip(confGrid, axis=2)

        #For horizontal flipping, only x-offset values must be adjusted to (1 - original offset)
        xOffsetGrid = np.squeeze(xOffsetGrid, axis=-1)
        xOffsetGridFlipped = np.flip(xOffsetGrid, axis=2)
        xOffsetGridFlippedAdjusted = np.where(xOffsetGridFlipped > 0.00, 1 - xOffsetGridFlipped, xOffsetGridFlipped)

        yOffsetGrid = np.squeeze(yOffsetGrid, axis=-1)
        yOffsetGridFlipped = np.flip(yOffsetGrid, axis=2)

        labelEntryAugmented = np.stack((confGridFlipped, xOffsetGridFlippedAdjusted, yOffsetGridFlipped,),axis=-1)

        return np.stack((dataEntry, dataEntryAugmented), axis=0).astype(np.float32), np.stack((labelEntry, labelEntryAugmented), axis=0).astype(np.float32)

    return dataEntry, labelEntry

#Randomly determines if an instance is a training instance or a validation instance
def isValInstance():
    return(random.random() < VAL_SPLIT) 



numMatchFolders = len([f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f)) and f.startswith("match")])

if(not numMatchFolders >= 1):
    print("\nERROR: No folders starting with 'match' were found in directory: " + str(INPUT_DIR))
    exit(1)

if(not os.path.exists(EXPORT_DIR)):
    os.mkdir(EXPORT_DIR)

trainFileName = EXPORT_DIR + '/train0.tfrecord'
valFileName = EXPORT_DIR + '/val.tfrecord'

trainWriter = tf.io.TFRecordWriter(trainFileName)
valWriter = tf.io.TFRecordWriter(valFileName)

#Loops through all match folders' images and annotations and stores generated instances in TFRecord files
for m in range(1, numMatchFolders+1):
    matchPath = os.path.join(INPUT_DIR, "match" + str(m))
    if(not os.path.exists(matchPath)):    
        print("\nERROR: The following directory does not exist: " + str(matchPath))
        exit(1)
    framesPath = os.path.join(matchPath, "frames")
    if(not os.path.exists(os.path.join(matchPath, 'Labels.csv'))):    
        print("\nERROR: No 'Labels.csv' file found at path: " + str(os.path.join(matchPath, 'Labels.csv')))
        exit(1)
    with open(os.path.join(matchPath, 'Labels.csv')) as csvfile:
        reader = csv.DictReader(csvfile)
        readerList = list(reader)
        i = 0
        while(i + IMGS_PER_INSTANCE - 1 < len(readerList)):
            currInstanceIndex = i
            visibilities = []
            xCoords = []
            yCoords = []
            imgs = []

            for j in range(0,IMGS_PER_INSTANCE):
                imgPath = os.path.join(framesPath, str(int(readerList[i+j]['Frame'])) + '.png')

                if(not os.path.exists(imgPath)):    
                    print("\nERROR: Image not found at path: " + str(imgPath))
                    exit(1)

                img = load_img(imgPath)
                img = img_to_array(img)

                visibilities.append(int(readerList[i+j]['Visibility']))
                xCoords.append(int(readerList[i+j]['X']))
                yCoords.append(int(readerList[i+j]['Y']))

                imgs.append(img)

            dataEntry, labelEntry = getDataAndLabels(imgs, xCoords, yCoords, visibilities)

            if(AUGMENT_DATA):
                #Adding the augmented version to train data.
                example = serializeExample(dataEntry[1], labelEntry[1])
                currBatchIdx += 1
                numTrainInstances += 1
                trainWriter.write(example)
                if(currBatchIdx == BATCH_SIZE):
                    batchCount += 1
                    trainFileName = EXPORT_DIR + '/train' + str(batchCount) + '.tfrecord'
                    trainWriter = tf.io.TFRecordWriter(trainFileName)
                    currBatchIdx = 0

                example = serializeExample(dataEntry[0], labelEntry[0])
            else:
                example = serializeExample(dataEntry, labelEntry)
            
            if(isValInstance()):
                numTestInstances += 1
                valWriter.write(example)
            else:
                currBatchIdx += 1
                numTrainInstances += 1
                trainWriter.write(example)
                if(currBatchIdx == BATCH_SIZE):
                    batchCount += 1
                    trainFileName = EXPORT_DIR + '/train' + str(batchCount) + '.tfrecord'
                    trainWriter = tf.io.TFRecordWriter(trainFileName)
                    currBatchIdx = 0
                
                                    
            #Prints progress bar
            percent_complete = int(round(m * 100 / (numMatchFolders+1), 0))
            bar = "#" * int(percent_complete / 10)
            space = " " * (10 - int(percent_complete / 10))
            print("\rProcessing Match {}: |{}{}| {}%. Training instances: {}, validation instances: {}.".format(m, bar, space, percent_complete,numTrainInstances,numTestInstances), end="")


            i = currInstanceIndex + NEXT_IMG_INDEX

print("\nDone.")




                        



                    

            



