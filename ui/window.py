# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from mainwindow import Ui_CarpalTunnelUNetPrediction

if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_CarpalTunnelUNetPrediction()

    ui.setupUi(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_()) 