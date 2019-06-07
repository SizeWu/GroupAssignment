import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QMessageBox, QAction, qApp, QMainWindow, QPushButton
from PyQt5.QtWidgets import QDesktopWidget, QAction, QMenu, QLabel, QLineEdit, QGridLayout
from PyQt5.QtGui import QIcon, QPainter
from PyQt5.QtCore import *

from Demo import Picture
from Demo import Picturestyle
from PyQt5.QtCore import *

# ----------------------------------
# 定义Example类
# ----------------------------------

class Example(QMainWindow,QWidget):

    def __init__(self):

        super().__init__()
        self.central_widget = QWidget()  # 建一个 central widget
        self.setCentralWidget(self.central_widget)  # 设置 QMainWindow.centralWidget
        self.initUI()

    def initUI(self):
        self.setWindowTitle('展示')  # 窗口名称
        self.resize(1350, 900)

        # ----------------------------------
        # 加入各个查询按钮
        # ----------------------------------
        self.bt1 = QPushButton('选择图片', self)# 选择图片按钮
        #self.bt1.setGeometry(10, 20, 250, 30)
        self.bt2 = QPushButton('添加颜色', self)  # 添加颜色按钮
        #self.bt2.setGeometry(280, 20, 250, 30)
        self.bt3 = QPushButton('风格迁移', self)  # 风格迁移按钮
        #self.bt3.setGeometry(550, 20, 250, 30)
        # ----------------------------------
        # 连接各个事件与信号槽
        # ----------------------------------
        self.bt1.clicked.connect(self.choiceimage)  # 选择图片信号
        self.bt2.clicked.connect(self.drawcolor)    # 添加颜色信号
        self.bt3.clicked.connect(self.formmove)     # 风格迁移信号
        # ----------------------------------
        # 在label上显示图片
        # ----------------------------------
        self.choicelable = QLabel(self)
        
        self.colorlabel = QLabel(self)         #添加颜色，显示图片的两个框
        self.graylabel = QLabel(self)
        self.anotherlabel = QLabel(self)             #风格迁移，显示图片的两个框
        self.formlabel = QLabel(self)

        grid = QGridLayout()                             # 界面布局
        self.centralWidget().setLayout(grid)
        grid.setSpacing(20)
        grid.addWidget(self.bt1, 1, 0)
        grid.addWidget(self.bt2, 1, 1)
        grid.addWidget(self.bt3, 1, 2)
        grid.addWidget(self.choicelable, 2, 0)
        grid.addWidget(self.graylabel, 2, 1)
        grid.addWidget(self.anotherlabel, 2, 2)
        grid.addWidget(self.colorlabel, 3, 1)
        grid.addWidget(self.formlabel, 3, 2)
        self.setLayout(grid)

        self.pic = ''                       # 记录图片的路径
        self.picstyle = ''

        self.show()
        self.center()

    # 询问是否确定退出
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认', '确认退出吗', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # 将页面显示在中央
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 选择图片函数
    def choiceimage(self):
        image_file, imgtype= QFileDialog.getOpenFileName(self, 'Open file', "C:\\", '*.jpg  *.png *.jpeg')
        self.pic = image_file
        print(self.pic)
        imagein = QImage(image_file)
        self.choicelable.setFixedWidth(400)                            # 固定图片大小（防止图片过大撑满界面）
        self.choicelable.setFixedHeight(320)
        self.choicelable.setStyleSheet("border: 2px solid red")                  # 为图片加上边框，设置边框属性
        result = imagein.scaled(self.choicelable.width(),self.choicelable.height(),Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
        self.choicelable.setPixmap(QPixmap.fromImage(result))       # 显示图片
        self.choicelable.setScaledContents(True)

    def drawcolor(self):                   #上面选完图片之后按按钮，会生成黑白和彩色图片
        if self.pic is not '':
            filegray, filecolor = Picture(self.pic)                # 返回保存的黑白和上色图片的路径
            print(filegray)
            print(filecolor)
            imagegray = QImage(filegray)                        # 根据路径打开上面的两张图片
            imagecolor = QImage(filecolor)
            self.graylabel.setFixedWidth(400)                     # 图片显示设置，与上面一样
            self.graylabel.setFixedHeight(320)
            self.graylabel.setStyleSheet("border: 2px solid red")
            result1 = imagegray.scaled(self.graylabel.width(), self.graylabel.height(), Qt.IgnoreAspectRatio,
                                    Qt.SmoothTransformation)
            self.graylabel.setPixmap(QPixmap.fromImage(result1))
            self.graylabel.setScaledContents(True)
            self.colorlabel.setFixedWidth(400)
            self.colorlabel.setFixedHeight(320)
            self.colorlabel.setStyleSheet("border: 2px solid red")
            result2 = imagecolor.scaled(self.colorlabel.width(), self.colorlabel.height(), Qt.IgnoreAspectRatio,
                                       Qt.SmoothTransformation)
            self.colorlabel.setPixmap(QPixmap.fromImage(result2))
            self.colorlabel.setScaledContents(True)
        else:
            pass


    def formmove(self):                     # 进行风格迁移的函数
        image_file, imgtype = QFileDialog.getOpenFileName(self, 'Open file', "C:\\", '*.jpg  *.png *.jpeg')
        self.picstyle = image_file
        if self.picstyle is not '':                            # 防止没有选中图片出bug
            fileform = Picturestyle(self.pic, self.picstyle)
            imagein = QImage(image_file)
            imageform = QImage(fileform)
            self.anotherlabel.setFixedWidth(400)
            self.anotherlabel.setFixedHeight(320)
            self.anotherlabel.setStyleSheet("border: 2px solid red")
            result1 = imagein.scaled(self.anotherlabel.width(), self.anotherlabel.height(), Qt.IgnoreAspectRatio,
                                       Qt.SmoothTransformation)
            self.anotherlabel.setPixmap(QPixmap.fromImage(result1))
            self.anotherlabel.setScaledContents(True)
            self.formlabel.setFixedWidth(400)
            self.formlabel.setFixedHeight(320)
            self.formlabel.setStyleSheet("border: 2px solid red")
            result2 = imageform.scaled(self.formlabel.width(), self.formlabel.height(), Qt.IgnoreAspectRatio,
                                     Qt.SmoothTransformation)
            self.formlabel.setPixmap(QPixmap.fromImage(result2))
            self.formlabel.setScaledContents(True)
            self.formlabel.setScaledContents(True)
        else:
            pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())