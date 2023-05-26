import cv2
import sys
import os
import argparse

from PySide2 import QtCore
from PySide2.QtCore import QDir, Qt, QRectF, QPoint
from PySide2.QtGui import QImage, QKeyEvent, QPainter, QPixmap, QColor, QPen, QFont
from PySide2.QtWidgets import QApplication, QLabel, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QWidget
from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton
from PySide2.QtWidgets import QFileDialog

parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

parser.add_argument('--video_dir', required=True, type=str, help="Path to .mp4 video file.")
parser.add_argument('--export_dir', required=True, type=str, help="Export directory where frames will be stored. Must be named with prefix 'match' following by an index.")

args = parser.parse_args()

VIDEO_DIR = args.video_dir
EXPORT_DIR = args.export_dir

def validDir(directory):
    suffix = directory.split('/')[-1]
    if not suffix.startswith('match'):  
        return False  
    number_part = suffix[5:] 
    return number_part.isdigit() 

if(not (validDir(EXPORT_DIR ))):
    print("\nERROR: Specified export folder does not start with 'match' followed by an index.")
    exit(1)

if(not os.path.exists(EXPORT_DIR)):
    os.mkdir(EXPORT_DIR)

frameDir = os.path.join(EXPORT_DIR, "frames")

if(not os.path.exists(frameDir)):
    os.mkdir(frameDir)

if(not (VIDEO_DIR.endswith('.mp4') and not VIDEO_DIR.startswith('.'))):
    print("\nERROR: Specified video is not in .mp4 format.")
    exit(1)

cap= cv2.VideoCapture(VIDEO_DIR)
i = 0
width = 1280
height = 720

numFrames = 0

try:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if(fps >= 57 and fps < 61):
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
        print("rendering " + str(numFrames) + " frames.")
    elif(fps >= 23 and fps <= 31):
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("rendering " + str(numFrames) + " frames.")
    else:
        raise ''
except:
    print("Video is not 60 fps and also not 30 fps")
    exit(1)
    

ret = cap.isOpened()
while(ret):
    percentage = i*100/numFrames
    print('Exporting...[%d%%]\r'%int(percentage), end="")
    if(fps>57 and fps < 61):
        #skip one frame
        ret, frame = cap.read()
    
    ret, frame = cap.read()

    if ret == False:
        break
    img = cv2.resize(frame, ( width , height ))
    cv2.imwrite(frameDir + "/" + str(i) + '.png', img)   
    i+=1

cap.release()
cv2.destroyAllWindows()

print("\nDone.")
quit()

