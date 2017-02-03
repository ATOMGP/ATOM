import sys
import pandas as pd
import multiprocessing

from PyQt4 import QtGui, uic, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class NewProject (QtGui.QDialog):

	def __init__(self):
		super(NewProject, self).__init__()
		uic.loadUi('frontend/newproject.ui', self)

		path = 'graphics/ATOM_icon-small.png'

		self.setWindowTitle("ATOM - New Project")
		self.setWindowIcon(QtGui.QIcon(path))
		self.init()

	def init(self):
		self.infoDict = {}
		self.browseProjectLocBtn.clicked.connect(self.select_dir_action)
		self.okBtn.clicked.connect(self.ok_action)
		self.cancelBtn.clicked.connect(self.close)


	def open_dir(self):
		dlg = QFileDialog()
		dlg.setFileMode(QFileDialog.Directory)
		if dlg.exec_():
			dirname = dlg.selectedFiles()
			return dirname[0]
		return ""


	def select_dir_action(self):
		self.projectLocTxt.setText(self.open_dir())

	def ok_action(self):
		valid = not (self.projectLocTxt.text() == ""
					or self.projectNameTxt.text() == "")
		if valid == True:
			self.infoDict['projectLoc'] = str(self.projectLocTxt.text())
			self.infoDict['projectName'] = str(self.projectNameTxt.text())
			self.close()

	def launch(self):
		self.infoDict = {}
		self.exec_()
		return self.infoDict
