from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, 
                            QWidget, QPushButton, QLabel, QComboBox, 
                            QVBoxLayout, QHBoxLayout, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSlot
import sys
from process_function import AppWindow

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        MW_Layout = QGridLayout() #set main window arrange mode
        MW_Layout.addWidget(self.Cifar10_Classifier_Groupbox(), 0, 0, 4, 1)
        self.setLayout(MW_Layout)

    def Cifar10_Classifier_Groupbox(self):
        AW = AppWindow()
        CCGroupbox = QGroupBox("5. Cifar10_Classifier")
        ShowImageButton = QPushButton("5.1 Show Train Images", self)
        ShowHyperButton = QPushButton("5.2 Show Hyperparameter", self)
        ShowModelButton = QPushButton("5.3 Show Model Structure", self)
        ShowAccuracyButton = QPushButton("5.4 Show Accuracy", self)
        TestGroupbox = QGroupBox("5.5 Test")
        ImageIndexGroupbox = QGroupBox("Image Index")
        ImageIndexSpinbox = QSpinBox()
        ImageIndexSpinbox.setMaximum(9999)
        TestButton = QPushButton("Test", self)

        ShowImageButton.clicked.connect(lambda:AW.Load_Cifar10_dataset_Func())
        ShowHyperButton.clicked.connect(lambda:AW.Load_Hyperparameter_Func())
        ShowModelButton.clicked.connect(lambda:AW.Show_Model_Structure_Func())
        ShowAccuracyButton.clicked.connect(lambda:AW.Show_Accuracy_and_Loss_Func())
        TestButton.clicked.connect(lambda:AW.Test_Model_Func(ImageIndexSpinbox.value()))

        II_Layout = QGridLayout()
        II_Layout.addWidget(ImageIndexSpinbox, 0, 0)
        ImageIndexGroupbox.setLayout(II_Layout)

        Test_Layout = QGridLayout()
        Test_Layout.addWidget(ImageIndexGroupbox, 0, 0)
        Test_Layout.addWidget(TestButton, 1, 0)
        TestGroupbox.setLayout(Test_Layout)

        CC_Layout = QGridLayout()
        CC_Layout.addWidget(ShowImageButton, 0, 0)
        CC_Layout.addWidget(ShowHyperButton, 1, 0)
        CC_Layout.addWidget(ShowModelButton, 2, 0)
        CC_Layout.addWidget(ShowAccuracyButton, 3, 0)
        CC_Layout.addWidget(TestGroupbox, 4, 0)
        CCGroupbox.setLayout(CC_Layout)
        return CCGroupbox  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())