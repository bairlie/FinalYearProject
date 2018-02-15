import sys, os, caffe
import numpy as np
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PIL.ImageQt import ImageQt
from PIL import Image
from SegmentImage import segmentImage
from AlphaChannelConfig import defAlpha

#qim = ImageQt(r'E:\Final Year Project\Python\2007_000392.jpg')

#Generate Transparent Image for Drawing on
img = np.zeros([500, 400,4],dtype=np.uint8)
img.fill(255)
im = Image.fromarray(img)
background = defAlpha(im, 0, 0)
background = ImageQt(background)

#Define Class Array
labels = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
brushColours = [QColor(255,255,255,10), QColor(128, 0, 0,10), QColor(0, 128,   0,10), QColor(128, 128,   0,10), QColor(  0,   0, 128,10), QColor(128,   0, 128,10), QColor(  0, 128, 128,10)
, QColor(128, 128, 128,100), QColor( 64,   0,   0,10), QColor(192,   0,   0,10), QColor( 64, 128,   0,10), QColor(192, 128,   0,10), QColor( 64, 0, 128,10), QColor(192,   0, 128,10)
, QColor(64, 128, 128,10), QColor(192, 128, 128,10), QColor(0,  64,   0,10), QColor(128,  64,   0,10), QColor(0, 192,   0,10), QColor(128, 192, 0,10), QColor(128, 192, 0,10), QColor(0,  64, 128,10)]
labels[0][0] = "Background"
labels[1][0] = brushColours[0]
for i in range(1,20):
    labels[0][i] = ("Class " + str(i))
    labels[1][i] = brushColours[i]

#Set-Up Window
class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 600, 500)
        self.setWindowTitle("Caffe Segmentation Tool")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
        self.pix = None
        self.home()
		
    def home(self):
        self.initActions()
    
        #Set-up Menubar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        optionMenu = mainMenu.addMenu("&Options")
        fileMenu.addAction(self.openAct)
        
        btn = QtGui.QPushButton("Run", self)
        btn.clicked.connect(self.updateImage)
        btn.resize(btn.minimumSizeHint())
        btn.move(0,50)
        
        saveBtn = QtGui.QPushButton("Save", self)
        saveBtn.clicked.connect(self.saveImage)
        saveBtn.resize(saveBtn.minimumSizeHint())
        saveBtn.move(100,50)
        
        #Add Dropdown Menu for Object Classes
        self.comboBox = QtGui.QComboBox(self)
        for i in range(0,20):
            self.comboBox.addItem(labels[0][i])
        self.comboBox.move(250, 50)
        self.comboBox.currentIndexChanged.connect(self.changeClass)
        
        #Add Textbox for class labelling
        self.textbox = QLineEdit(self)
        self.textbox.move(400, 50)
        self.textbox.resize(200,30)
        
        classBtn = QtGui.QPushButton("Label", self)
        classBtn.clicked.connect(self.updateClasses)
        classBtn.resize(classBtn.minimumSizeHint())
        classBtn.move(400,90)
        
        #Set-up variables for drawing
        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 10
        self.myPenColor = brushColours[0]
        self.lastPoint = QtCore.QPoint()
        
        #Set-Up Image Pixmaps and Painter
        self.pic = QtGui.QLabel(self)
        self.pic.setGeometry(50, 10, 600, 500)
        self.segment = QtGui.QLabel(self)
        self.segment.setGeometry(50, 50, 600, 500)
        self.segs = QtGui.QPixmap.fromImage(background)
        self.segment.setPixmap(QtGui.QPixmap(self.segs))
        self.painter = QtGui.QPainter(self.segs)
        if(self.pix != None):
            self.pic.setPixmap(QtGui.QPixmap(self.pix))
        self.show()
        
    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File",
            QtCore.QDir.currentPath())
        if fileName:
            loadedImage = QtGui.QImage()
            if not loadedImage.load(fileName):
                return False
            self.pix = QtGui.QPixmap.fromImage(loadedImage)
            self.pic.setPixmap(QtGui.QPixmap(self.pix))
            self.pic.update()
            
    def initActions(self):
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
            triggered=self.open)
        #self.extractAction = QtGui.QAction("&GET TO THE CHOPPAH!!!", self)
        #self.extractAction.setShortcut("Ctrl+Q")
        #self.extractAction.setStatusTip('Leave The App')
        #self.extractAction.triggered.connect(self.close)
    
    def changeClass(self):
        for i in range(0,20):
            if(str(self.comboBox.currentText()) == labels[0][i]):
                self.myPenColor = labels[1][i]
        
    
    def updateClasses(self):
        for i in range(0,20):
            if(str(self.comboBox.currentText()) == labels[0][i]):
                labels[0][i] = self.textbox.text()
        self.comboBox.clear()
        
        for i in range(0,20):
            self.comboBox.addItem(labels[0][i])
        self.comboBox.update()
    
    #Perform Image Segmentation, then overlay segments over original image
    def updateImage(self):
        overlay = segmentImage('deploy.prototxt', 'fcn2s-dilated-vgg19.caffemodel', '2007_000738.jpg')
        overlay = defAlpha(overlay, 100, 0)
        overlay = ImageQt(overlay)
        self.segs = QtGui.QPixmap.fromImage(overlay)    
        self.segment.setPixmap(QtGui.QPixmap(self.segs))
        #self.painter = QtGui.QPainter(self.segs)
        #self.painter.end()
        self.segment.show()
    
    def saveImage(self):
        self.segs.save('test.png')
        img = Image.open('test.png')
        segMasks = defAlpha(img, 255, 255)
        segMasks.save('test.png')
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.lastPoint.setY(self.lastPoint.y() - 85)
            self.lastPoint.setX(self.lastPoint.x() - 50)
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False
           
    def drawLineTo(self, endPoint):
        endPoint.setY(endPoint.y() - 85)
        endPoint.setX(endPoint.x() - 50)
        painter = QtGui.QPainter(self.segs)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth,
            QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        painter.end()
        modified = True
        self.segment.setPixmap(QtGui.QPixmap(self.segs))
        self.segment.update()
        self.lastPoint = QtCore.QPoint(endPoint)
				
#Launch Window		
def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

   
run()

