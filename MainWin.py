import sys
import time
import os
import threading
from manager import *
from shared import *

from PyQt4 import QtGui, uic
from PyQt4.QtGui import QFileDialog
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from NewProject import NewProject
from settings import SettingsWin
import multiprocessing as mp

class MainWin (QtGui.QMainWindow):

    def __init__(self):
        super(MainWin, self).__init__()
        uic.loadUi('frontend/main-win2.ui', self)
        
        path = 'graphics/ATOM_icon-small.png'

        self.setWindowTitle("ATOM")
        self.setWindowIcon(QtGui.QIcon(path))
        
        self.init()
        self.set_img()
        self.show()

    def set_img(self):
        palette = QPalette()
        palette.setBrush(QPalette.Background,QBrush(QPixmap("graphics/ATOM_Welcome_Small.png")))
        self.setPalette(palette)

    def init(self):
        self.final_dic ={}
        self.manager_thread = []

        self.timer = QTimer()        
        self.last = ('',0)
        
        self.shared_mem = mp.Queue()
        self.pause = True

        self.progress_val = 0;
        self.first_new = 0

        self.outputText.setTextBackgroundColor(QColor(0,0,0,255))
        self.outputText.setTextColor(QColor(16,175,35,255))
        self.outputText.setFontPointSize(10)
        self.outputText.setFontWeight(1)

        self.settings = SettingsWin()

        self.newProjectBasic = {}
        self.newProjectParam = {}
        
        self.progress.setVisible(False)
        self.pauseResumeBtn.setVisible(False)
        self.cancelBtn.setVisible(False)
        self.showDetailsCheck.setVisible(False)
        self.outputText.setVisible(False)
        self.openReportBtn.setVisible(False)

        
        self.runBtn.setEnabled(False)
        self.settingsBtn.setEnabled(False)
        
        # menu buttons.
        self.newProjectBtn.triggered.connect(self.new_project_action)
        self.openProjectBtn.triggered.connect(self.open_project_action)
        self.saveBtn.triggered.connect(self.save_project_action)
        self.exitBtn.triggered.connect(self.exit)
        self.docBtn.triggered.connect(self.show_doc)
        self.aboutBtn.triggered.connect(self.show_about)
        self.runBtn.triggered.connect(self.run_action)
        self.settingsBtn.triggered.connect(self.settings_action)
        self.showDetailsCheck.toggled.connect(self.show_det_check_action)
        self.timer.timeout.connect(self.update_UI)
        self.pauseResumeBtn.clicked.connect(self.pause_resume_action)
        self.openReportBtn.clicked.connect(self.openRep)
        self.cancelBtn.clicked.connect(self.cancelAction)

        self.timer.start(300)




    def cancelAction(self):
        self.manager_thread.terminate()
        self.outputText.append('job aborted')

    def openRep(self):
        os.system('xdg-open "example.pdf"')

    def update_UI(self):
        while self.shared_mem.empty() is False:
            res = self.shared_mem.get_nowait()
            tmp = res.read()
            if tmp[1] != -1:
                self.progress.setValue(tmp[1])
            self.outputText.append(tmp[0])
            if tmp[0] == 'generated final report':
                self.progress.setValue(100)
                self.openReportBtn.setVisible(True)
                self.cancelBtn.setEnabled(False)
                self.runBtn.setEnabled(True)



    def run_action(self):
        self.pauseResumeBtn.setVisible(True)
        self.progress.setVisible(True)
        self.showDetailsCheck.setVisible(True)
        self.cancelBtn.setVisible(True)

        self.runBtn.setEnabled(False)
        self.settingsBtn.setEnabled(False)
        
        self.start_task(True)

    

    def show_det_check_action(self):
        self.outputText.setVisible(self.showDetailsCheck.isChecked())

    def start_task(self, tmp=True):
        if tmp is True:
            self.final_dic = dict(self.newProjectBasic.items() + self.newProjectParam.items())
            
        self.manager = Manager(self.final_dic)

        self.manager_thread = mp.Process(target=self.manager.run, args = (self.shared_mem,))
        
        self.manager_thread.start()

        self.setPalette(QApplication.palette())

    
    def new_project_action(self):
        newProject = NewProject()
        tmp = newProject.launch()
        if len(tmp) != 0:
            self.newProjectBasic = tmp
            self.settingsBtn.setEnabled(True)
            self.settings_action(self.first_new!=0)
            self.first_new = 1
    
    def settings_action(self, reset=False):
        print reset
        if reset == True:
            self.settings = SettingsWin()
        tmp = self.settings.launch()
        if not (tmp['trainPath'] == '' or (tmp['testPath'] == '' and tmp['oneDataset'] == False)):
            self.newProjectParam = tmp
            self.runBtn.setEnabled(True)


    
    def select_file_action(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter("ATOMPROJ files (*.atomproj)")
        filenames = []
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            return filenames[0]
        return ""

    def save_project_action(self):
        pickle.dump(self.final_dic, open( self.final_dic['projectLoc'] + '/' +self.final_dic['projectName'] + '.atomproj', "w+" ))

    def show_about(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("ATOM aims at making machine leaning tasks easy and fast by automating it.")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_doc(self):
        os.system('xdg-open "../document/document.pdf"')
        pass

    def open_project_action(self):
        filename = self.select_file_action()
        self.final_dic = pickle.load(open(filename, 'rb'))
        self.run_action(False)


    def pause_resume_action(self):
        if self.pause is True:
            self.manager_thread.terminate()
            self.pause = False
            self.pauseResumeBtn.setText('Resume')
        else:
            self.start_task()
            self.pause = True
            self.pauseResumeBtn.setText('Pause')

    
    def exit(self):
        self.close()

    # override original virtual close_event
    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit ATOM?"
        reply = QtGui.QMessageBox.question(self, 'Message', 
                quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            self.manager_thread.terminate()
            event.accept()
        else:
            event.ignore()

app = QtGui.QApplication(sys.argv)
GUI = MainWin()
sys.exit(app.exec_())
