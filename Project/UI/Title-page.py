import sys
from PyQt6.QtWebEngineWidgets import QWebEngineView

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from Main_page import HTMLViewerApp

class StartupWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RSSS')
        self.setMinimumSize(500,500)
        # self.setMaximumSize(1000,1000)
        parentLayout = QGridLayout()
        
        
        
        #設計
        self.setStyleSheet('''
            QMainWindow{
                background-color: #CCFFCC;
            }
            QPushButton{
                background-color: QLinearGradient(x0: 0, y0: 0, x1: 1, y1: 1, stop: 0 #0093E9, stop: 1 #80D0C7);
                padding: 10px;
                border: none;
                border-radius: 5px;
                margin: 10px 0 0 0;
                color: white;
            }
            *{
                color: #403f3e;
                font-family: "Montserrat";
                font-size: 12px;
                font-weight: 700;
            }
            #title_label{
                font-size: 25px;
                color: #0093E9;
            }
            #lic_label{
                font-size: 5px;
                color: #0093E9;
            }
            QLineEdit{
                padding: 5px 2.5px;
                border-radius: 5px;
            }
        ''')
        #title
        title_label = QLabel('Insight survey interface')
        title_label.setObjectName("title_label")
        
        #tips
        tips_button = QPushButton('Tips')
        tips_button.setIcon(QIcon('./icon/Bookmark.png'))
        tips_button.clicked.connect(self.open_tips_window)
        
        self.new_window = None
        
        
        #lic
        # lic_label = QLabel('Doc')
        lic_button = QPushButton('Docs')
        lic_button.setIcon(QIcon('./icon/Document.png'))
        # lic_label.setObjectName("lic_label")
        
        #Image
        # img_label = QLabel()
        # pixmap = QPixmap('image.png')
        # img_label.setPixmap(pixmap)
        # img_label.adjustSize()
        # img_label.move(0,0)
        
        
        #Student ID
        SID_label = QLabel("Name:")
        Grade_label = QLabel('Grade:')
        SID_input = QLineEdit()
        
        #Grade
        Grade_combo = QComboBox()
        Grade_combo.addItems(['Choose your Grade!','M1','M2','D1','D2','D3','PD'])
        Grade_combo.setCurrentText('Choose your Grade!')
    

        #Start
        self.Start_button = QPushButton('Start')
        self.Start_button.clicked.connect(self.open_main_window)
        # self.Start_button.clicked.connect(self.open_o_window)
        
        
        parentLayout.addWidget(lic_button, 0, 0,alignment=Qt.AlignmentFlag.AlignTop)
        # parentLayout.addWidget(img_label, 0, 0,alignment=Qt.AlignmentFlag.AlignTop.AlignLeft)
        parentLayout.addWidget(tips_button, 0, 1, alignment=Qt.AlignmentFlag.AlignTop.AlignRight)
        parentLayout.addWidget(title_label, 1, 1, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter.AlignLeft)
        parentLayout.addWidget(SID_label, 2, 0, alignment=Qt.AlignmentFlag.AlignLeft.AlignRight)
        parentLayout.addWidget(Grade_label, 3, 0, alignment=Qt.AlignmentFlag.AlignLeft.AlignRight)
        parentLayout.addWidget(SID_input, 2, 1)
        parentLayout.addWidget(Grade_combo, 3, 1)
        parentLayout.addWidget(self.Start_button, 4, 0, 1, 2)
        
        centerWidget = QWidget()
        centerWidget.setLayout(parentLayout)
        self.setCentralWidget(centerWidget)
    
    def open_main_window(self):
        self.main_window = HTMLViewerApp()
        self.main_window.show()
        self.close()

    
    def open_tips_window(self):
        if self.new_window is None:
            self.new_window = TipsWindow()
        self.new_window.show()
    
    # quit message
    # def closeEvent(self,e):
    #     yesno = QMessageBox.question(self,'quit','Ensure you want to leave?')
    #     if yesno == QMessageBox.StandardButton.Yes:
    #         e.accept()
    #     else:
    #         e.ignore()

class TipsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200,200)
        self.setWindowTitle('Tips')
        Layout = QVBoxLayout()
        self.setLayout(Layout)
        
        tips_text = QLabel('Main function of this system \n\n 1. Create a knowledge interface for insights survey assistants from multiple academic papers on a specific research topic. \n 2. From the origin (root) of the research task, expand the citation inheritance and relevance associations. \n 3. Explore the relevance chain within similar research tasks to highlight key research points.')
        Layout.addWidget(tips_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    startup_window = StartupWindow()
    startup_window.show()
    sys.exit(app.exec())
