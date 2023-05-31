import sys
import os
import csv
import argparse

from PySide2 import QtCore
from PySide2.QtCore import QDir, Qt, QRectF, QPoint, QPointF
from PySide2.QtGui import QImage, QKeyEvent, QPainter, QPixmap, QColor, QPen, QFont, QBrush, QTransform
from PySide2.QtWidgets import QApplication, QLabel, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QWidget
from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QGroupBox
from PySide2.QtWidgets import QFileDialog, QGraphicsTextItem
from PySide2.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

parser.add_argument('--match_dir', required=True, type=str, help="Match directory. Must be named with prefix 'match' following by an index.")

args = parser.parse_args()

MATCH_DIR = args.match_dir
FRAMES_DIR = os.path.join(MATCH_DIR, "frames")
CSV_DIR = os.path.join(MATCH_DIR, "Labels.csv")

if(not os.path.exists(MATCH_DIR)):
    print("\nERROR: The following directory does not exist: " + str(MATCH_DIR))
    exit(1)

def validDir(directory):
    _, suffix = os.path.split(directory)
    if not suffix.startswith('match'):  
        return False  
    number_part = suffix[5:] 
    return number_part.isdigit()

if(not (validDir(MATCH_DIR))):
    print("\nERROR: Specified export folder does not start with 'match' followed by an index.")
    exit(1)

if(not os.path.exists(FRAMES_DIR)):
    print("\nERROR: The following directory does not exist: " + str(FRAMES_DIR))
    exit(1)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.view = QGraphicsView()
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setRenderHints(QPainter.HighQualityAntialiasing | QPainter.SmoothPixmapTransform | QPainter.NonCosmeticDefaultPen)
        self.view.setMouseTracking(True)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setRenderHint(QPainter.NonCosmeticDefaultPen, True)
        
        self.view.wheelEvent = self.wheelEvent
        self.view.mousePressEvent = self.getPixelCoordinates
        
        self.images = []
        self.frameIndex = 0
        self.pixelCoordinates = {}
        self.states = {}
        self.annotated = {}
        self.currentCenterPoint = self.view.mapToScene(self.view.viewport().rect().center()) 
        self.loadImages()

        self.pixmap = QPixmap(self.images[self.frameIndex])
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.pixmap)
        self.view.setScene(self.scene)

        self.zoomLevel = 1.0

        self.toggleStateButton = QPushButton("Toggle State", self)
        self.toggleStateButton.setFixedSize(200, 75)
        self.toggleStateButton.clicked.connect(self.toggleState)

        self.removePixelButton = QPushButton("Remove Pixel", self)
        self.removePixelButton.setFixedSize(200, 75)
        self.removePixelButton.clicked.connect(self.removePixel)

        self.removeFrameButton = QPushButton("Remove Frame", self)
        self.removeFrameButton.setFixedSize(200, 75)
        self.removeFrameButton.clicked.connect(self.removeFrame)

        self.saveResultsButton = QPushButton("Save Results", self)
        self.saveResultsButton.setFixedSize(200, 75)
        self.saveResultsButton.clicked.connect(self.saveResults)

        self.visbilityText = QLabel()
        self.visbilityText.setStyleSheet("color: red;")
        self.visbilityText.setFont(QFont("Arial", 30))

        self.imageText = QLabel()
        self.imageText.setStyleSheet("color: black;")
        self.imageText.setFont(QFont("Arial", 30))

        self.annotatedText = QLabel()   
        self.annotatedText.setFont(QFont("Arial", 30))
        
        self.topLayout = QHBoxLayout()
        self.topLayout.addWidget(self.visbilityText)
        self.topLayout.addWidget(self.imageText)
        self.topLayout.addWidget(self.annotatedText)

        self.buttonLayout = QVBoxLayout()
        self.buttonLayout.addWidget(self.toggleStateButton)
        self.buttonLayout.addWidget(self.removePixelButton)
        self.buttonLayout.addWidget(self.removeFrameButton)
        self.buttonLayout.addWidget(self.saveResultsButton)

        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addWidget(self.view)
        self.bottomLayout.addLayout(self.buttonLayout)
        self.bottomLayout.setAlignment(self.buttonLayout, Qt.AlignRight)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.addLayout(self.bottomLayout)

    
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

        self.showImage()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0: 
            self.zoomLevel = self.zoomLevel * 1.2
            self.view.scale(1.2, 1.2)
        else:
            self.zoomLevel = self.zoomLevel / 1.2
            self.view.scale(1 / 1.2, 1 / 1.2)
        return True
        

    def loadImages(self):
        global FRAMES_DIR 
        directory = QDir(FRAMES_DIR)
        directory.setNameFilters(["*.png"])
        directory.setSorting(QDir.Name)
        self.images = [directory.filePath(file) for file in directory.entryList()]
        self.images = sorted(self.images, key=lambda x: int(x.split('/')[-1].split(".")[0]))

        for i in range (len(self.images)):
            self.annotated[i] = False

    def showImage(self):
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setRenderHint(QPainter.HighQualityAntialiasing)
        self.view.setRenderHint(QPainter.TextAntialiasing)
        self.view.setRenderHint(QPainter.NonCosmeticDefaultPen)

        self.pixmap = QPixmap(self.images[self.frameIndex])
        self.view.setScene(QGraphicsScene())
        self.view.scene().addPixmap(self.pixmap)
        self.view.fitInView(self.view.sceneRect(), Qt.KeepAspectRatio)
        self.view.scale(self.zoomLevel, self.zoomLevel)
        self.view.centerOn(self.currentCenterPoint)

        self.imageText.setText(self.images[self.frameIndex].split('/')[-1])

        if not self.frameIndex in self.states:
            self.states[self.frameIndex] = "VISIBLE"

        self.visbilityText.setText(self.states[self.frameIndex])

        if(self.frameIndex in self.pixelCoordinates or self.states[self.frameIndex] == "INVISIBLE"):
            self.annotatedText.setText("Annotated")
            self.annotatedText.setStyleSheet("color: green;")
        else:
            self.annotatedText.setText("Not Annotated")
            self.annotatedText.setStyleSheet("color: red;")


        if(self.frameIndex in self.pixelCoordinates):
            x, y = self.pixelCoordinates[self.frameIndex]
            dotSize = 1
            self.view.scene().addEllipse(x-dotSize/2, y-dotSize/2, dotSize, dotSize, QPen(Qt.red))
   
    def toggleState(self):
        if(self.frameIndex in self.states):
            if(self.states[self.frameIndex] == "VISIBLE"):
                self.states[self.frameIndex]  = "INVISIBLE"
            else:
                self.states[self.frameIndex]  = "VISIBLE"

        if(self.states[self.frameIndex]  == "INVISIBLE"):
            self.annotated[self.frameIndex] = True
        elif(self.states[self.frameIndex]  == "VISIBLE" and self.frameIndex not in self.pixelCoordinates):
            self.annotated[self.frameIndex] = False
        
        self.showImage()

    def removePixel(self):
        if self.frameIndex in self.pixelCoordinates:
            del self.pixelCoordinates[self.frameIndex]
            if(self.states[self.frameIndex] == "VISIBLE"):
                self.annotated[self.frameIndex] = False
            self.showImage()

    def removeFrame(self):
        print("Removed image: " + str(self.images[self.frameIndex]))

        if(self.frameIndex in self.pixelCoordinates):
            self.pixelCoordinates.pop(self.frameIndex)
            for key in list(self.pixelCoordinates.keys()):
                if key > self.frameIndex:
                    value = self.pixelCoordinates.pop(key)
                    self.pixelCoordinates[key-1] = value

        if(self.frameIndex in self.states):    
            self.states.pop(self.frameIndex)
            for key in list(self.states.keys()):
                if key > self.frameIndex:
                    value = self.states.pop(key)
                    self.states[key-1] = value

        if(self.frameIndex in self.annotated):    
            self.annotated.pop(self.frameIndex)
            for key in list(self.annotated.keys()):
                if key > self.frameIndex:
                    value = self.annotated.pop(key)
                    self.annotated[key-1] = value
        
        os.remove(self.images[self.frameIndex])
        
        self.images.pop(self.frameIndex)

        if(self.frameIndex >= len(self.images)):
            self.frameIndex = len(self.images) - 1

        self.showImage()

    def saveResults(self):
        global FRAMES_DIR

        allAnnotated = True

        for i in range(len(self.images)):
            if(not self.annotated[i]):
                allAnnotated = False
                print(str(self.images[i].split('/')[-1]) + " is not annotated.")

        if(allAnnotated):
            with open(CSV_DIR, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "Visibility", "X", "Y"])
                indices = list(range(0,len(self.pixelCoordinates)))
    
                data = []   
                for i in range(len(self.images)):
                    if(self.states[i] == "VISIBLE"):
                        visibility = 1
                        
                    else:
                        visibility = 0
                    
                    if(i in self.pixelCoordinates):
                        x_coord = int(self.pixelCoordinates[i][0])
                        y_coord = int(self.pixelCoordinates[i][1])
                    else:
                        x_coord = 0
                        y_coord = 0
                        
                    data.append([i, visibility, x_coord, y_coord])

                writer.writerows(data)
                print("Saved Results!")
            
            for i in range(len(self.images)):
                os.rename(str(self.images[i]), os.path.join(FRAMES_DIR, str(i) + ".png"))

            sys.exit(app.exec_())
      
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_A:
            if self.frameIndex > 0:
                self.frameIndex -= 1
                self.showImage()
        elif event.key() == Qt.Key_D:
            if self.frameIndex < len(self.images) - 1:
                self.frameIndex += 1
                self.showImage()

    def getPixelCoordinates(self, event):
        pos = event.pos()
        scene_pos = self.view.mapToScene(pos)

        self.pixelCoordinates[self.frameIndex] = (scene_pos.x(), scene_pos.y())
        self.annotated[self.frameIndex] = True

        dotSize = 1
        self.view.scene().addEllipse(scene_pos.x()-dotSize/2, scene_pos.y()-dotSize/2, dotSize, dotSize, QPen(Qt.red))
        
        self.currentCenterPoint = QPointF(scene_pos.x(), scene_pos.y())
        if self.frameIndex < len(self.images) - 1:
            self.frameIndex += 1
            self.showImage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    #viewer.showFullScreen()
    sys.exit(app.exec_())
