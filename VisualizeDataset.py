import sys
import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QWidget
from PySide2.QtGui import QPixmap, QImage, QFont
from PySide2.QtCore import Qt
import tensorflow as tf
import argparse
import time
import cv2
import os
from glob import glob

WIDTH = 768
HEIGHT = 432

GRID_COLS = 48
GRID_ROWS = 27

IMGS_PER_INSTANCE = 5

parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

parser.add_argument('--data_dir', required=True, type=str, help="Data directory containing the .tfrecord files.")

args = parser.parse_args()

DATA_DIR = args.data_dir

def _parse_function(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(example['image'], tf.float32)
    label = tf.io.decode_raw(example['label'], tf.float32)

    image = tf.reshape(image, (IMGS_PER_INSTANCE*3, HEIGHT, WIDTH))
    label = tf.reshape(label, (IMGS_PER_INSTANCE, GRID_ROWS, GRID_COLS, 3))
    return image, label

def load_data_and_labels_from_tfrecord(tfrecord_file):
    print(tfrecord_file)
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(_parse_function)
    data, labels = [], []
    for image, label in dataset:
        data.append(image.numpy())
        labels.append(label.numpy())
    x_data = np.stack(data)
    y_data = np.array(labels)
    return x_data, y_data


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.currentInstanceIndex = 0
        self.indexInInstance = 0
        self.currentFileIndex = 0
        self.tfrecord_file_pattern = os.path.join(DATA_DIR,"train*.tfrecord")
        self.numFiles = len(glob(self.tfrecord_file_pattern)) 
        self.currentFile = os.path.join(DATA_DIR, "train" + str(self.currentFileIndex) + ".tfrecord")
        self.x_data, self.y_data = load_data_and_labels_from_tfrecord(self.currentFile)

        self.image_widget = QLabel()

        self.fileIndex = QLabel()
        self.fileIndex.setStyleSheet("color: black;")
        self.fileIndex.setFont(QFont("Arial", 20))

        self.instanceIndex = QLabel()
        self.instanceIndex.setStyleSheet("color: black;")
        self.instanceIndex.setFont(QFont("Arial", 20))

        self.frameIndex = QLabel()   
        self.frameIndex.setFont(QFont("Arial", 20))
        
        self.text_layout = QVBoxLayout()
        self.text_layout.addWidget(self.fileIndex)
        self.text_layout.addWidget(self.instanceIndex)
        self.text_layout.addWidget(self.frameIndex)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.text_layout)
        self.main_layout.addWidget(self.image_widget)

        self.nav_layout = QVBoxLayout()
        self.prevFile_button = QPushButton("Previous File")
        self.nextFile_button = QPushButton("Next File")
        self.prevFile_button.clicked.connect(self.previous_file)
        self.nextFile_button.clicked.connect(self.next_file)
        self.nav_layout.addWidget(self.prevFile_button)
        self.nav_layout.addWidget(self.nextFile_button)

        self.prev_button = QPushButton("Previous Instance")
        self.next_button = QPushButton("Next Instance")
        self.prev_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.next_button)

        self.prevFrame_button = QPushButton("Previous Frame")
        self.nextFrame_button = QPushButton("Next Frame")
        self.prevFrame_button.clicked.connect(self.previousFrame)
        self.nextFrame_button.clicked.connect(self.nextFrame_frame)
        self.nav_layout.addWidget(self.prevFrame_button)
        self.nav_layout.addWidget(self.nextFrame_button)

        self.main_layout.addLayout(self.nav_layout)
       

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.set_image_and_heatmap()

    def set_image_and_heatmap(self):
        self.fileIndex.setText("File \n" + str(self.currentFileIndex+1) + "/" + str(self.numFiles))
        self.instanceIndex.setText("Instance \n" + str(self.currentInstanceIndex+1) + "/" + str(self.y_data.shape[0]))
        self.frameIndex.setText("Frame \n" + str(self.indexInInstance+1) + "/"  + str(self.y_data.shape[1]))

        image = self.x_data[self.currentInstanceIndex]
        image = image[self.indexInInstance*3:self.indexInInstance*3+3].copy()
        image = np.moveaxis(image,0,-1).copy()
        y_true = self.y_data[self.currentInstanceIndex][self.indexInInstance].copy()

        heatmap, x, y  = np.split(y_true,3,axis=-1)    
        heatmap = np.squeeze(heatmap)
        x_offset = np.squeeze(x)
        y_offset = np.squeeze(y)

        indexMax = np.argmax(heatmap)
        row, col = np.unravel_index(indexMax, heatmap.shape)

        image = (image * 255).astype(np.uint8)

        heatmap = (heatmap*255).astype(np.uint8)
        heatmap = np.repeat(np.repeat(heatmap, HEIGHT // GRID_ROWS, axis=0), WIDTH // GRID_COLS, axis=1)
        heatmap = np.expand_dims(heatmap,axis=-1)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

        step_x = heatmap.shape[1] // GRID_COLS
        step_y = heatmap.shape[0] // GRID_ROWS

        for x in range(0, heatmap.shape[1] + 1, step_x):
            cv2.line(heatmap, (x, 0), (x, heatmap.shape[0]), (255, 255, 255), 1)

        for y in range(0, heatmap.shape[0] + 1, step_y):
            cv2.line(heatmap, (0, y), (heatmap.shape[1], y), (255, 255, 255), 1)
        
        ball = (int(((col + x_offset[row,col])* WIDTH)/GRID_COLS ),int(((row + y_offset[row,col])* HEIGHT)/GRID_ROWS ))
        cv2.circle(image, ball, 3, (255, 255, 0), 1)

        alpha = 0.8
        beta = (1.0 - alpha)
        heatmap = cv2.addWeighted(image, alpha, heatmap, beta, 0.0)

        #qheatmap = QImage(np.ascontiguousarray(heatmap).data, heatmap.shape[1], heatmap.shape[0], QImage.Format_RGB888)
        #self.heatmap_widget.setPixmap(QPixmap.fromImage(qheatmap))
        qimage = QImage(np.ascontiguousarray(heatmap).data, heatmap.shape[1], heatmap.shape[0], QImage.Format_RGB888)
        self.image_widget.setPixmap(QPixmap.fromImage(qimage))

    def previous_file(self):  
        if self.currentFileIndex > 0:
            self.currentFileIndex -= 1
            self.currentFile = os.path.join(DATA_DIR, "train" + str(self.currentFileIndex) + ".tfrecord")
            self.x_data, self.y_data = load_data_and_labels_from_tfrecord(self.currentFile)
            self.currentInstanceIndex = 0
            self.indexInInstance = 0
            self.set_image_and_heatmap()
    
    def next_file(self):  
        if self.currentFileIndex < self.numFiles - 1:
            self.currentFileIndex += 1
            self.currentFile = os.path.join(DATA_DIR, "train" + str(self.currentFileIndex) + ".tfrecord")
            self.x_data, self.y_data = load_data_and_labels_from_tfrecord(self.currentFile)
            self.currentInstanceIndex = 0
            self.indexInInstance = 0
            self.set_image_and_heatmap()

    def previous_frame(self):
        if self.currentInstanceIndex > 0:
            self.currentInstanceIndex -= 1
            self.indexInInstance=0
            self.set_image_and_heatmap()

    
    def previousFrame(self):
        if self.indexInInstance > 0:
            self.indexInInstance -= 1
            self.set_image_and_heatmap()

    def nextFrame_frame(self):
        if self.indexInInstance < self.y_data.shape[1] - 1:
            self.indexInInstance += 1
            self.set_image_and_heatmap()
    
    def next_frame(self):
        if self.currentInstanceIndex < self.x_data.shape[0] - 1:
            self.currentInstanceIndex += 1
            self.indexInInstance=0

            self.set_image_and_heatmap()    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())