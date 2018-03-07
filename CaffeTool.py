#############################################################################
##
## Copyright (C) 2010 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

import sys, os, io, caffe
import numpy as np
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PIL.ImageQt import ImageQt
import cStringIO
from PIL import Image
from SegmentImage import segmentImage
from AlphaChannelConfig import defAlpha
#from CheckBox import ExampleCombobox
from GMMSegmentation import segViaGMM

#qim = ImageQt(r'E:\Final Year Project\Python\2007_000392.jpg')

#Generate Transparent Image for Drawing on
img = np.zeros([500, 500,4],dtype=np.uint8)
img.fill(255)
im = Image.fromarray(img)
background = defAlpha(im, 1, 1)
background = ImageQt(background)

#Define Class Array
labels = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
brushColours = [QColor(255,255,255,255), QColor(128, 0, 0,255), QColor(0, 128,   0,255), QColor(128, 128,   0,255), QColor(  0,   0, 128,255), QColor(128,   0, 128,255), QColor(  0, 128, 128,255)
, QColor(128, 128, 128,255), QColor( 64,   0,   0,255), QColor(192,   0,   0,255), QColor( 64, 128,   0,255), QColor(192, 128,   0,255), QColor( 64, 0, 128,255), QColor(192,   0, 128,255)
, QColor(64, 128, 128,255), QColor(192, 128, 128,255), QColor(0,  64,   0,255), QColor(128,  64,   0,255), QColor(0, 192,   0,255), QColor(128, 192, 0,255), QColor(128, 192, 0,255), QColor(0,  64, 128,255)]
labels[0][0] = "Background"
labels[1][0] = brushColours[0]
for i in range(1,20):
    labels[0][i] = ("Class " + str(i))
    labels[1][i] = brushColours[i]

#Set-Up Window
class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 600, 600)
        self.setWindowTitle("Caffe Segmentation Tool")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
        self.pix = None
        self.imgName = None
        self.home()

    def home(self):
        self.initActions()

        #Set-up Menubar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        optionMenu = mainMenu.addMenu("&Options")
        optionMenu.addAction(self.penWidthAct)
        #optionMenu.addAction(self.segModelAct)
        optionMenu.addAction(self.toggleGMMAct)
        fileMenu.addAction(self.openAct)
        fileMenu.addAction(self.saveFile)
        fileMenu.addAction(self.segImage)

        #Initialise buffer for converting Pixmaps to PIL images
        self.buffer = QBuffer()
        self.buffer.open(QIODevice.ReadWrite)

        #Add Dropdown Menu for Object Classes
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItem(labels[0][0])
        self.comboBox.setItemData(0, QBrush(QColor(0,0,0,255)), Qt.TextColorRole)
        for i in range(1,20):
            self.comboBox.addItem(labels[0][i])
            self.comboBox.setItemData(i, QBrush(labels[1][i]), Qt.TextColorRole)
        self.comboBox.move(0, 25)
        self.comboBox.currentIndexChanged.connect(self.changeClass)

        #Add Textbox for class labelling
        self.textbox = QLineEdit(self)
        self.textbox.move(120, 25)
        self.textbox.resize(200,30)

        #Add label button
        classBtn = QtGui.QPushButton("Label", self)
        classBtn.clicked.connect(self.updateClasses)
        classBtn.resize(classBtn.minimumSizeHint())
        classBtn.move(340,25)

        #Set-up segmentation variables
        self.segModelProto = 'deploy.prototxt'
        self.segModelCaffe = 'fcn2s-dilated-vgg19.caffemodel'

        #Set-up variables for drawing
        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 10
        self.myPenColor = brushColours[0]
        self.lastPoint = QtCore.QPoint()

        #Set-Up GMM Tool Variables
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.gmmActive = False #Testing Variable - for now

        #Set-Up Image Pixmaps and Painter
        self.pic = QtGui.QLabel(self)
        self.pic.setGeometry(0, 55, 600, 500)
        self.segment = QtGui.QLabel(self)
        self.segment.setGeometry(0, 55, 600, 500)
        self.segs = QtGui.QPixmap.fromImage(background)
        self.segment.setPixmap(QtGui.QPixmap(self.segs))
        #self.painter = QtGui.QPainter(self.segs)
        if(self.pix != None):
            self.pic.setPixmap(QtGui.QPixmap(self.pix))
        self.show()

    #Method for changing the segmentation model - Uncomment
    #def changeModel(self):
    #        dialog = ExampleCombobox()
    #        answer = dialog.exec_()
    #        if answer[0] == int(QMessageBox.Ok):
    #            print (str(answer[1]))

    #Method for opening an image file
    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File",
            QtCore.QDir.currentPath())
        if fileName:
            loadedImage = QtGui.QImage()
            if not loadedImage.load(fileName):
                return False
            self.imgName = str(fileName)
            self.pix = QtGui.QPixmap.fromImage(loadedImage)
            self.pic.setPixmap(QtGui.QPixmap(self.pix))
            w = self.pix.width()
            h = self.pix.height()
            self.pic.setGeometry(0, 60, w, h)
            self.segment.setGeometry(0, 60, w, h)
            self.pic.update()

    #Method for saving an image file TODO: Add JSON export
    def file_save(self):
        #files_types = ["Text files (*.txt)", "Images (*.png *.jpg)"]
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        self.segs.save(name)
        dictionary = dict(zip(labels[0], labels[1]))
        #img = Image.open(name)
        #segMasks = defAlpha(img, 255, 255)
        #segMasks.save(name)

    #Method for turning the GMM tool on and off
    def toggleGMM(self):
        self.gmmActive = not self.gmmActive

    #Method for chaning the brush width
    def penWidth(self):
        newWidth, ok = QtGui.QInputDialog.getInteger(self, "app",
            "Select pen width:", self.myPenWidth, 1, 50, 1)
        if(ok):
            self.myPenWidth = newWidth

    #Initialise Actions for Menubar
    def initActions(self):
        #Open Image
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
            triggered=self.open)

        #Save Image
        self.saveFile = QtGui.QAction("&Save File", self)
        self.saveFile.setShortcut("Ctrl+S")
        self.saveFile.setStatusTip('Save File')
        self.saveFile.triggered.connect(self.file_save)

        #Run Segmentation
        self.segImage = QtGui.QAction("&Segment", self, shortcut="Ctrl+R", triggered=self.updateImage)

        #Change Pen Width
        self.penWidthAct = QtGui.QAction("Pen &Width...", self,
            triggered=self.penWidth)

        #Toggle Gaussian Mixture Model Tool
        self.toggleGMMAct = QtGui.QAction("Toggle &GMM", self,
            triggered=self.toggleGMM)

        #Change Segmentation Model
        #self.segModelAct = QtGui.QAction("Change &Model...", self,
        #    triggered=self.changeModel)

    #Set Brush Colour based on the selected Class
    def changeClass(self):
        for i in range(0,20):
            if(str(self.comboBox.currentText()) == labels[0][i]):
                self.myPenColor = labels[1][i]

    #Update Class labels
    def updateClasses(self):
        for i in range(0,20):
            if(str(self.comboBox.currentText()) == labels[0][i]):
                labels[0][i] = self.textbox.text()
        self.comboBox.clear()

        self.comboBox.addItem(labels[0][0])
        self.comboBox.setItemData(0, QBrush(QColor(0,0,0,255)), Qt.TextColorRole)
        for i in range(1,20):
            self.comboBox.addItem(labels[0][i])
            self.comboBox.setItemData(i, QBrush(labels[1][i]), Qt.TextColorRole)
        self.comboBox.update()

    #Perform Image Segmentation, then overlay segments over original image
    def updateImage(self):
        if(self.imgName != None):
            overlay = segmentImage(self.segModelProto, self.segModelCaffe ,self.imgName)
            overlay = defAlpha(overlay, 100, 1)
            overlay = ImageQt(overlay)
            self.segs = QtGui.QPixmap.fromImage(overlay)
            self.segment.setPixmap(QtGui.QPixmap(self.segs))
            #self.painter = QtGui.QPainter(self.segs)
            self.segment.show()

    #Method for Detecting a Mouse Press
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.gmmActive:
                self.origin = QPoint(event.pos())
                #self.origin.setY(self.origin.y() - 60)
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
            else:
                self.lastPoint = event.pos()
                self.lastPoint.setY(self.lastPoint.y() - 60)
                #self.lastPoint.setX(self.lastPoint.x() - 50)
                self.scribbling = True

    #Method for Detecting Mouse Movement
    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton):
            if self.gmmActive and not self.origin.isNull():
                self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())
            elif self.scribbling:
                newPoint = event.pos()
                newPoint.setY(newPoint.y() - 60)
                self.drawLineTo(event.pos())

    #Method for Detecting Mouse Button Release
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            #Code for when the GMM Tool is active
            if self.gmmActive:
                self.rubberBand.hide()
                endPoint = event.pos()
                minX = min(self.origin.x(), endPoint.x())
                maxX = max(self.origin.x(), endPoint.x())
                minY = min(self.origin.y(), endPoint.y()) - 60
                maxY = max(self.origin.y(), endPoint.y()) - 60
                crop_rectangle = (minX, minY, maxX, maxY)

                #Save/Load Image to Buffer, used to convert from QImage to PIL Image
                img = Image.open(self.imgName)
                cropped_im = img.crop(crop_rectangle)
                cropCol = cropped_im.load()
                segLayer = QImage(self.segs)
                segLayer.save(self.buffer, "PNG")
                strio = cStringIO.StringIO()
                strio.write(self.buffer.data())
                self.buffer.close()
                strio.seek(0)
                mask = Image.open(strio)
                maskCrop = mask.crop(crop_rectangle)
                maskCrop = maskCrop.convert('RGB')
                maskCol = maskCrop.load()
                cropY = max(0,(self.origin.y() - 61) - minY)
                cropX = max(0,((self.origin.x() - 1) - minX))
                if len(maskCrop.getcolors()) > 1:
                    if maskCol[cropX,cropY][:3] == maskCrop.getcolors()[0][1]:
                        gmmSeg = segViaGMM(cropped_im, maskCol[cropX,cropY][:3] , maskCrop.getcolors()[1][1][:3], cropX, cropY)
                    else :
                        gmmSeg = segViaGMM(cropped_im, maskCol[cropX,cropY][:3] , maskCrop.getcolors()[0][1][:3], cropX, cropY)
                else:
                    gmmSeg = segViaGMM(cropped_im, maskCrop.getcolors()[0][1][:3], maskCrop.getcolors()[0][1][:3], cropX, cropY)
                self.updateMask(gmmSeg, minX, maxX, minY, maxY)
            elif self.scribbling:
                self.drawLineTo(event.pos())
                self.scribbling = False
                newSegLayer = QImage(self.segs)
                newSegLayer.save(self.buffer, "PNG")
                strio = cStringIO.StringIO()
                strio.write(self.buffer.data())
                self.buffer.close()
                strio.seek(0)
                segMasks = Image.open(strio)
                segMasks = defAlpha(segMasks, 100, 1)
                layer = ImageQt(segMasks)
                self.segs = QtGui.QPixmap.fromImage(layer)
                self.segment.setPixmap(QtGui.QPixmap(self.segs))
                self.segment.update()

    #Method used by GMM tool to update the segmentation mask with GMM tool output
    def updateMask(self, crop, xmin, xmax, ymin, ymax):
        newSegLayer = QImage(self.segs)
        newSegLayer.save(self.buffer, "PNG")
        strio = cStringIO.StringIO()
        strio.write(self.buffer.data())
        self.buffer.close()
        strio.seek(0)
        maskLayer = Image.open(strio)
        maskLayerRGB = maskLayer.convert('RGB')
        maskLayerRGB = np.array(maskLayerRGB)
        crop = np.array(crop)
        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                maskLayerRGB[y,x] = crop[y - ymin, x - xmin]
        newMask = Image.fromarray(maskLayerRGB)
        newMask = defAlpha(newMask, 100, 1)
        layer = ImageQt(newMask)
        self.segs = QtGui.QPixmap.fromImage(layer)
        self.segment.setPixmap(QtGui.QPixmap(self.segs))
        self.segment.update()

    #Method for undoing the last action TODO: Complete Function
    def undoAction(self):
        self.lastMask

    #Method for drawing a line between 2 points
    def drawLineTo(self, endPoint):
        endPoint.setY(endPoint.y() - 60)
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
