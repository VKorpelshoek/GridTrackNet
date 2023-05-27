import cv2
import argparse
import os
import numpy as np
import tensorflow as tf
from GridTrackNet import GridTrackNet

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH/GRID_COLS
GRID_SIZE_ROW = HEIGHT/GRID_ROWS

MODEL_DIR = os.path.join(os.getcwd(),"model_weights.h5")
model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
model.load_weights(MODEL_DIR)

def getPredictions(frames, isBGRFormat = False):
    outputHeight = frames[0].shape[0]
    outputWidth = frames[0].shape[1]
    
    batches = []
    for i in range(0, len(frames), 5):
        batch = frames[i:i+5]
        if len(batch) == 5:
            batches.append(batch)

    units = []
    for batch in batches:
        unit = []
        for frame in batch:
            if(isBGRFormat):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(WIDTH,HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, -1, 0)
            unit.append(frame[0])
            unit.append(frame[1])
            unit.append(frame[2])
        units.append(unit) 
    
    units = np.asarray(units)
    units = units.astype(np.float32)
    units /= 255

    y_pred = model.predict(units, batch_size=len(batches),verbose=0)
    
    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    confGrid, xOffsetGrid, yOffsetGrid = np.split(y_pred, 3, axis=-1)

    confGrid = np.squeeze(confGrid, axis=-1)
    xOffsetGrid = np.squeeze(xOffsetGrid, axis=-1)
    yOffsetGrid = np.squeeze(yOffsetGrid, axis=-1)

    ballCoordinates = []
    for i in range(0, confGrid.shape[0]):
        for j in range(0, confGrid.shape[1]):
            currConfGrid = confGrid[i][j]
            currXOffsetGrid = xOffsetGrid[i][j]
            currYOffsetGrid = yOffsetGrid[i][j]

            maxConfVal = np.max(currConfGrid)
            predRow, predCol = np.unravel_index(np.argmax(currConfGrid), currConfGrid.shape)

            threshold = 0.5
            predHasBall = maxConfVal >= threshold

            xOffset = currXOffsetGrid[predRow][predCol]
            yOffset = currYOffsetGrid[predRow][predCol]

            xPred = int((xOffset + predCol) * GRID_SIZE_COL)
            yPred = int((yOffset + predRow) * GRID_SIZE_ROW)

            if(predHasBall):
                ballCoordinates.append((int((xPred/WIDTH)*outputWidth), int((yPred/HEIGHT)*outputHeight)))
            else:
                ballCoordinates.append((0,0))

    return ballCoordinates


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

    parser.add_argument('--video_dir', required=True, type=str, help="Path to .mp4 video file.")
    parser.add_argument('--model_dir', required=False, default=os.path.join(os.getcwd(),"model_weights.h5"), type=str, help="Path to saved Tensorflow model.")
    parser.add_argument('--display_trail', required=False, default=1, type=int, help="Output a visible trail of the ball's trajectory. Default = 1")

    args = parser.parse_args()

    VIDEO_DIR = args.video_dir
    MODEL_DIR = args.model_dir 
    DISPLAY_TRAIL = bool(args.display_trail)

    model.load_weights(MODEL_DIR)

    cap = cv2.VideoCapture(VIDEO_DIR)

    if not cap.isOpened():
        print("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if(fps >= 57 and fps <= 62):
        numFramesSkip = 2
    elif (fps >= 22 and fps <= 32):
        numFramesSkip = 1
    else:
        print("ERROR: Video is not 30FPS or 60FPS")
        exit(0)

    totalFrames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // numFramesSkip)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  

    directory, filename = os.path.split(VIDEO_DIR)
    name, extension = os.path.splitext(filename)
    output_filename = name + " Predicted" + extension
    output_path = os.path.join(directory, output_filename)

    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    index = 0

    frames = []
    ballCoordinatesHistory = []
    numPredicted = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("\nDone")
            break
        
        if(index % numFramesSkip == 0):
            numPredicted += 1
            frames.append(frame)
            if(len(frames) == 5):
                ballCoordinates = getPredictions(frames, True)
                for ball in ballCoordinates:
                    ballCoordinatesHistory.append(ball)
                    
                for i, frame in enumerate(frames):
                    if(i < len(ballCoordinates)):
                        if(DISPLAY_TRAIL):
                            if(len(ballCoordinatesHistory) >= 15):
                                for j in range(7,-1,-2):
                                    idx = len(ballCoordinatesHistory)-5-j+i
                                    cv2.circle(frame, ballCoordinatesHistory[idx], 4, (0, 255, 255),-1)
                        else:
                            cv2.circle(frame, ballCoordinates[i], 8, (0, 0, 255),4)

                    video_writer.write(frame)

                frames = []


        index += 1
        percentage = numPredicted*100/totalFrames
        print('Exporting...[%d%%]\r'%int(percentage), end="")

    cap.release()
    video_writer.release()

    cv2.destroyAllWindows()