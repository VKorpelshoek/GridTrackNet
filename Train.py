import numpy as np
import sys, getopt
import os
from glob import glob
from keras.models import *
from keras.layers import *
from GridTrackNet import GridTrackNet
import keras.backend as K
from keras import optimizers
from keras.activations import *
import tensorflow as tf
import math
from keras.losses import *
import argparse
import csv

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

IMGS_PER_INSTANCE = 5
HEIGHT = 432
WIDTH = 768
GRID_COLS = 48
GRID_ROWS = 27

parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

parser.add_argument('--data_dir', required=True, type=str, help="Data directory of the folder containing all folders with names with the prefix 'match'.")
parser.add_argument('--load_weights', required=False, type=str, help="Directory to load pre-trained weights.")
parser.add_argument('--save_weights', required=True, type=str, help="Directory to store model weights and training metrics.")
parser.add_argument('--epochs', required=True, type=int, help='Number of epochs (iterations of the training data) the model should be trained for.')
parser.add_argument('--tol', required=False, type=int, default=4, help='Specifies the tolerance of the model: the number of pixels the predicted location is allowed to deviate from the true location. Default = 4')
parser.add_argument('--batch_size', required=False, type=int, default=5,help="Specify the batch size to train on. Default = 5")

args = parser.parse_args()

DATA_DIR = args.data_dir
LOAD_WEIGHTS = args.load_weights
SAVE_WEIGHTS = args.save_weights
EPOCHS = args.epochs
TOL = args.tol
BATCH_SIZE = args.batch_size

def calcOutcomeStats(y_pred, y_true):
	TP = TN = FP1 = FP2 = FN = 0

	#Reformat the 15 output grids into 5x3 grids.
	y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
	y_pred = np.stack(y_pred, axis=2)
	y_pred = np.moveaxis(y_pred, 1, -1)

	#Separate confGrid, xOffsetGrid, and yOffsetGrid
	confGridTrue, xOffsetGridTrue, yOffsetGridTrue = np.split(y_true, 3, axis=-1)
	confGridPred, xOffsetGridPred, yOffsetGridPred = np.split(y_pred, 3, axis=-1)

	#Remove last axis of all grids
	confGridTrue = np.squeeze(confGridTrue, axis=-1)
	xOffsetGridTrue = np.squeeze(xOffsetGridTrue, axis=-1)
	yOffsetGridTrue = np.squeeze(yOffsetGridTrue, axis=-1)
	confGridPred = np.squeeze(confGridPred, axis=-1)
	xOffsetGridPred = np.squeeze(xOffsetGridPred, axis=-1)
	yOffsetGridPred = np.squeeze(yOffsetGridPred, axis=-1)

	#Loop over instances in batch
	for i in range(0, confGridTrue.shape[0]):
		#Loop over frames in instance
		for j in range(0, confGridTrue.shape[1]):
			currConfGridTrue = confGridTrue[i][j]
			currXOffsetGridTrue = xOffsetGridTrue[i][j]
			currYOffsetGridTrue = yOffsetGridTrue[i][j]

			currConfGridPred = confGridPred[i][j]
			currXOffsetGridPred = xOffsetGridPred[i][j]
			currYOffsetGridPred = yOffsetGridPred[i][j]

			maxConfValTrue = np.max(currConfGridTrue)
			trueRow, trueCol = np.unravel_index(np.argmax(currConfGridTrue), currConfGridTrue.shape)

			maxConfValPred = np.max(currConfGridPred)
			predRow, predCol = np.unravel_index(np.argmax(currConfGridPred), currConfGridPred.shape)

			threshold = 0.5
			trueHasBall = maxConfValTrue >= threshold
			predHasBall = maxConfValPred >= threshold

			xOffsetTrue = currXOffsetGridTrue[trueRow][trueCol]
			yOffsetTrue = currYOffsetGridTrue[trueRow][trueCol]

			xOffsetPred = currXOffsetGridPred[predRow][predCol]
			yOffsetPred = currYOffsetGridPred[predRow][predCol]

			GRID_SIZE_COL = WIDTH/GRID_COLS
			GRID_SIZE_ROW = HEIGHT/GRID_ROWS

			#True Coordinates
			xTrue = int((xOffsetTrue + trueCol) * GRID_SIZE_COL)
			yTrue = int((yOffsetTrue + trueRow) * GRID_SIZE_ROW)

			#Predicted Coordinates
			xPred = int((xOffsetPred + predCol) * GRID_SIZE_COL)
			yPred = int((yOffsetPred + predRow) * GRID_SIZE_ROW)
			
			if ((not predHasBall) and (not trueHasBall)):
				TN += 1

			elif (predHasBall and (not trueHasBall)):
				FP2 += 1

			elif ((not predHasBall) and trueHasBall):
				FN += 1

			elif (predHasBall and trueHasBall):
				dist = int(((xPred - xTrue)**2 + (yPred - yTrue)**2)**0.5)

				if dist > TOL:
					FP1 += 1

				else:
					TP += 1

	return (TP, TN, FP1, FP2, FN)

GLOBAL_ACCURACY = 0
GLOBAL_PRECISION = 0
GLOBAL_RECALL = 0
GLOBAL_F1 = 0

#Helper function to compute training metrics from the statistics
def accuracy(y_true, y_pred):
	global GLOBAL_ACCURACY
	global GLOBAL_PRECISION
	global GLOBAL_RECALL
	global GLOBAL_F1

	(TP, TN, FP1, FP2, FN) = calcOutcomeStats(y_pred, y_true)
	try:
		GLOBAL_ACCURACY = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
	except:
		GLOBAL_ACCURACY = 0.0
	try:
		GLOBAL_PRECISION = TP / (TP + FP1 + FP2)
	except:
		GLOBAL_PRECISION = 0.0
	try:
		GLOBAL_RECALL = TP / (TP + FN)
	except:
		GLOBAL_RECALL = 0.0
	try:
		GLOBAL_F1 = (2*(GLOBAL_PRECISION*GLOBAL_RECALL))/(GLOBAL_PRECISION + GLOBAL_RECALL)
	except:
		GLOBAL_F1 = 0.0	
		
	return GLOBAL_ACCURACY
	
#Helper function to return computed metric
def precision(y_true, y_pred):
	global GLOBAL_PRECISION
	return GLOBAL_PRECISION

#Helper function to return computed metric
def recall(y_true, y_pred):
	global GLOBAL_RECALL
	return GLOBAL_RECALL

#Helper function to return computed metric
def f1(y_true, y_pred):
	global GLOBAL_F1
	return GLOBAL_F1

#Loss function consists of:
#1) L1 loss: 		x-offset/y-offset regression loss
#2) Focal loss: 	confidence grid loss, where contribution of 'easy' 
#					examples to the loss are downweighted by gamma parameter.
def custom_loss(y_true, y_pred):
	#Reformat the 15 output grids into 5x3 grids.
	y_pred = tf.split(y_pred, IMGS_PER_INSTANCE, axis=1)
	y_pred = tf.stack(y_pred, axis=2)
	y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])

	#Separate confGrid from offset grids
	confGridTrue, xOffsetGridTrue, yOffsetGridTrue = tf.split(y_true, 3, axis=-1)
	confGridPred, xOffsetGridPred, yOffsetGridPred = tf.split(y_pred, 3, axis=-1)

	#Combine x-offset and y-offset into single tensor
	yTrueOffset = tf.concat([xOffsetGridTrue, yOffsetGridTrue],axis=-1)
	yPredOffset = tf.concat([xOffsetGridPred, yOffsetGridPred],axis=-1)

	#Offset loss
	diff = tf.abs(yTrueOffset - yPredOffset)
	sum_diff = tf.reduce_sum(diff, axis=-1, keepdims=True)
	masked_sum_diff = confGridTrue * sum_diff	#Only compute loss for cell where confGridTrue = 1
	offset = tf.reduce_sum(masked_sum_diff, axis=[1,2,3,4])

	#Confidence loss (focal)
	gamma = 2
	positiveConfLoss = confGridTrue * tf.pow(1 - confGridPred, gamma) * tf.math.log(tf.clip_by_value(confGridPred, tf.keras.backend.epsilon(), 1))
	negativeConfLoss = (1 - confGridTrue) * tf.pow(confGridPred, gamma) *  tf.math.log(tf.clip_by_value(1 - confGridPred, tf.keras.backend.epsilon(), 1))
	confidence = tf.reduce_sum((-1)*(positiveConfLoss + negativeConfLoss),axis=[1,2,3,4])

	loss = offset + confidence

	return tf.reduce_sum(loss)

#Helper function to convert raw TFRecord file data entries into instances with labels
def parseInstance(rawData):
	feature_description = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.string),
	}
	example = tf.io.parse_single_example(rawData, feature_description)
	image = tf.io.decode_raw(example['image'], tf.float32)
	label = tf.io.decode_raw(example['label'], tf.float32)
	image = tf.reshape(image, (IMGS_PER_INSTANCE*3, HEIGHT, WIDTH))
	label = tf.reshape(label, (IMGS_PER_INSTANCE, GRID_ROWS, GRID_COLS, 3))
	return image, label

#Load TFRecord file
def loadSubDataset(file):
	subdataset = tf.data.TFRecordDataset(file)
	subdataset = subdataset.map(parseInstance, num_parallel_calls=tf.data.AUTOTUNE)
	return subdataset

#Loads all training files and generates a randomized dataset for each epoch
def createEpochDataset(tfRecordFile, bufferSize, batch_size, numParallelCalls):
	filenames = tf.data.Dataset.list_files(tfRecordFile, shuffle=True)

	interleavedDataset = filenames.interleave(
		loadSubDataset,
		cycle_length=numParallelCalls,
		num_parallel_calls=tf.data.AUTOTUNE,
		deterministic=False
	)

	dataset = interleavedDataset.shuffle(bufferSize)
	dataset = dataset.batch(batch_size)

	return dataset



#Loading model
ADADELTA = optimizers.Adadelta(learning_rate=1.0)
model = GridTrackNet(IMGS_PER_INSTANCE,HEIGHT, WIDTH)
model.compile(loss=custom_loss, optimizer=ADADELTA, metrics=[accuracy,precision,recall,f1],run_eagerly=True)

#Loading weights if specified in --load_weights argument
if not (args.load_weights is None):
	model.load_weights(LOAD_WEIGHTS)

print(model.summary())

if not os.path.exists(SAVE_WEIGHTS):
	os.makedirs(SAVE_WEIGHTS)

rawValData = tf.data.TFRecordDataset(os.path.join(DATA_DIR,"val.tfrecord"))#
parsedValData = rawValData.map(parseInstance)
valData = parsedValData.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

tfRecordFilePattern = os.path.join(DATA_DIR,"train*.tfrecord")

numTrainFiles = len(glob(tfRecordFilePattern)) - 1
numInstancesPerFile = 50
bufferSize = numInstancesPerFile  #Set to number of instances per batch for maximum randomness, 50 by default in DataGen.py
numParallelCalls = 8  # Adjust the number of parallel calls based on system's available resources
stepsPerEpoch = (numTrainFiles * numInstancesPerFile) // BATCH_SIZE

#Creates a new .csv file for outputting training and evaluation results
header = ['epoch', 'loss', 'val loss', 'accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall', 'f1', 'val_f1']
csvPath = SAVE_WEIGHTS + "/Results.csv"
if not os.path.exists(csvPath):
	f = open(csvPath, 'w', newline='')
	writer = csv.writer(f)
	writer.writerow(header)
	f.flush()
else:
	f = open(csvPath, 'a', newline='')
	writer = csv.writer(f)

#Loop for training model EPOCHS times. Evaluates model after each epoch and writes results to .csv file
for epoch in range(EPOCHS):
	print(f"\nTraining epoch {epoch + 1}/{EPOCHS}")
	dataset = createEpochDataset(tfRecordFilePattern, bufferSize, BATCH_SIZE, numParallelCalls)

	history = model.fit(dataset, epochs=1, steps_per_epoch=stepsPerEpoch, verbose=1)

	model_save_path = os.path.join(SAVE_WEIGHTS, "epoch_" + str(epoch+1) + ".h5")
	model.save_weights(model_save_path)

	values = list(history.history.values())

	print(f"\nEvaluating epoch {epoch + 1}/{EPOCHS}")
	valLoss, valAccuracy, valPrecision, valRecall, valF1 = model.evaluate(valData, verbose=1)

	writer.writerow([epoch+1, round(values[0][0], 6), round(valLoss, 6), round(values[1][0], 6), round(valAccuracy, 6), round(values[2][0], 6), round(valPrecision, 6), round(values[3][0], 6), round(valRecall, 6), round(values[4][0], 6), round(valF1, 6),])
	f.flush()

print('\nDone.')
f.close()