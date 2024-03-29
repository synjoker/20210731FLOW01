# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Desktop\EndNoteHere\researchHere\BOS\BOS-PIV\program\20210731FLOW01\Mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_open = QtWidgets.QPushButton(self.centralwidget)
        self.button_open.setGeometry(QtCore.QRect(120, 210, 75, 23))
        self.button_open.setObjectName("button_open")
        self.label_video = QtWidgets.QLabel(self.centralwidget)
        self.label_video.setGeometry(QtCore.QRect(310, 130, 371, 341))
        self.button_close = QtWidgets.QPushButton(self.centralwidget)
        self.button_close.setGeometry(QtCore.QRect(120, 260, 75, 23))
        self.button_close.setObjectName("button_close")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_open.setText(_translate("MainWindow", "open"))
        self.label_video.setText(_translate("MainWindow", "video"))
        self.button_close.setText(_translate("MainWindow", "close"))

    def slot_init(self):
        self.button_open.clicked.connect(self.button_open_clicked)  # 若该按键被点击，则调用button_open_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
 
    '''槽函数之一'''
 
    def button_open_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(50)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_video.clear()  # 清空视频显示区域
            self.button_open.setText('打开相机')
 
    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        self.image = cv2.flip(self.image, 1)
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_video.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
