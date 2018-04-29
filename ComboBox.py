import sys
from PySide.QtCore import *
from PySide.QtGui import *

class DialogWithComboBox(QMessageBox):

    def __init__(self, parent= None):
        super(DialogWithComboBox, self).__init__()

        self.comboBox = QComboBox(self)
        #Access the Layout of the MessageBox to add the Checkbox
        layout = self.layout()
        layout.addWidget(self.comboBox, 0,3)

    def exec_(self, *args, **kwargs):
        """
        Override the exec_ method so you can return the value of the checkbox
        """
        return QMessageBox.exec_(self, *args, **kwargs), self.comboBox.currentText()

class ExampleCombobox(DialogWithComboBox):

    def __init__(self, parent=None):
        super(ExampleCombobox, self).__init__()
        self.setWindowTitle("Change Segmentation Model")
        self.setText("Select Segmentation Model")
        self.comboBox.addItem("fcn2s-dilated-vgg19")
        self.comboBox.addItem("voc-fcn8s")
        self.comboBox.addItem("CSAIL-FCN")
        self.setStandardButtons(QMessageBox.Cancel |QMessageBox.Ok)
        self.setDefaultButton(QMessageBox.Cancel)
