import sys
import pandas as pd
import multiprocessing

from PyQt4 import QtGui, uic, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# TO DO : Acrivate Cancel Button

class SettingsWin (QtGui.QDialog):

	def __init__(self):
		super(SettingsWin, self).__init__()
		uic.loadUi('frontend/settings3.ui', self)

		path = 'graphics/ATOM_icon-small.png'

		self.setWindowTitle("ATOM - Settings")
		self.setWindowIcon(QtGui.QIcon(path))

		self.init()

	def init(self):
		self.config_dict = {}

		self.set_actions()
		self.tab_one_init()
		self.tab_two_init()
		self.tab_three_init()
		self.tab_four_init()
		self.tab_five_init()

		self.tab_one_save()
		self.tab_two_save()
		self.tab_three_save()
		self.tab_four_save()
		self.tab_five_save()

	def set_actions(self):

		# tab1 actions
		self.oneDatasetRadio.toggled.connect(self.one_dataset_toggled)
		self.browseTrainPathBtn.clicked.connect(self.browse_train_path)
		self.browseTestPathBtn.clicked.connect(self.browse_test_path)
		self.regressionCheck.toggled.connect(self.reg_config)
		self.tab1ApplyBtn.clicked.connect(self.tab_one_apply_action)
		self.tab1OkBtn.clicked.connect(self.tab_one_ok_action)
		self.tab1CancelBtn.clicked.connect(self.close)
		self.tab1RestoreBtn.clicked.connect(self.tab_one_restore_action)

		# tab2 actions
		self.tab2ApplyBtn.clicked.connect(self.tab_two_apply_action)
		self.tab2OkBtn.clicked.connect(self.tab_two_ok_action)
		self.tab2CancelBtn.clicked.connect(self.close)
		self.tab2RestoreBtn.clicked.connect(self.tab_two_restore_action)
		self.doFP.toggled.connect(self.tab_two_doFPAction)

		# tab3 actions
		self.tab3ApplyBtn.clicked.connect(self.tab_three_apply_action)
		self.tab3OkBtn.clicked.connect(self.tab_three_ok_action)
		self.tab3CancelBtn.clicked.connect(self.close)
		self.tab3RestoreBtn.clicked.connect(self.tab_three_restore_action)

		# tab4 actions
		self.tab4ApplyBtn.clicked.connect(self.tab_four_apply_action)
		self.tab4OkBtn.clicked.connect(self.tab_four_ok_action)
		self.tab4CancelBtn.clicked.connect(self.close)
		self.tab4RestoreBtn.clicked.connect(self.tab_four_restore_action)

		# tab5 actions
		self.tab5ApplyBtn.clicked.connect(self.tab_five_apply_action)
		self.tab5OkBtn.clicked.connect(self.tab_five_ok_action)
		self.tab5CancelBtn.clicked.connect(self.close)
		self.tab5RestoreBtn.clicked.connect(self.tab_five_restore_action)


	#----------------------- Tab1 -----------------------#

	def tab_one_init(self):
		self.targetCB.clear()
		self.testAndTrainRadio.setChecked(True)
		self.regressionCheck.setChecked(False)
		self.testRatioBox.setEnabled(False)
		self.testRatioBox.setRange(0.1, 0.4)
		self.testRatioBox.setSingleStep(0.01)
		self.MLPCheck.setVisible(False)
		self.MLPLbl.setVisible(False)
		self.numEvalMLP.setVisible(False)
		self.buildFinalCheck.setVisible(False)

	def one_dataset_toggled(self):
		self.set_one_dataset(not self.oneDatasetRadio.isChecked())


	def set_one_dataset(self, enable):
		self.testPathLbl.setEnabled(enable)
		self.testPathTxt.setEnabled(enable)
		self.browseTestPathBtn.setEnabled(enable)
		self.testRatioLbl.setEnabled(not enable)
		self.testRatioBox.setEnabled(not enable)

	def select_file_action(self):
		dlg = QFileDialog()
		dlg.setFileMode(QFileDialog.AnyFile)
		dlg.setFilter("CSV files (*.csv)")
		filenames = []
		if dlg.exec_():
			filenames = dlg.selectedFiles()
			return filenames[0]
		return ""

	def browse_train_path(self):
		path = self.select_file_action()
		self.trainPathTxt.setText(path)
		if path != "":
			self.set_target()


	def browse_test_path(self):
		self.testPathTxt.setText(self.select_file_action())


	def set_target(self):
		self.targetCB.clear()
		data = pd.read_csv(str(self.trainPathTxt.text()))
		self.targetCB.addItems(data.columns)


	def tab_one_ok_action(self):
		ret = self.tab_one_validate()
		if ret==True:
			self.tab_one_save()
			self.close()
		else:
			self.error_msg()

	def tab_one_apply_action(self):
		ret = self.tab_one_validate()
		if ret == False:
			self.error_msg()
		else:
			self.tab_one_save()

	def tab_one_restore_action(self):
		self.tab_one_init()
		pass

	def tab_one_save(self):

		self.config_dict['oneDataset'] = self.oneDatasetRadio.isChecked()
		self.config_dict['testRatio'] = float(self.testRatioBox.text())
		self.config_dict['trainPath'] = str(self.trainPathTxt.text())
		self.config_dict['testPath'] = str(self.testPathTxt.text())
		self.config_dict['target'] = str(self.targetCB.currentText())
		self.config_dict['regression'] = self.regressionCheck.isChecked()
		self.config_dict['verbose'] = self.verboseCheck.isChecked()

	def tab_one_validate(self):
		return not (self.trainPathTxt.text() == "")

	#----------------------- Tab2 -----------------------#

	def tab_two_init(self):

		self.handleMissingCheck.setEnabled(False)
		self.oneHotCheck.setEnabled(False)
		self.normCheck.setEnabled(False)
		self.logTransCheck.setChecked(True)
		self.pweCheck.setChecked(True)
		self.twoWayCheck.setChecked(False)
		self.threeWayCheck.setChecked(False)
		self.pweBox.setRange(0.0, 1.0)
		self.pweBox.setValue(1.0)
		self.pweBox.setSingleStep(0.01)

	
	def tab_two_doFPAction(self):
		self.logTransCheck.setEnabled(self.doFP.isChecked())
		self.pweCheck.setEnabled(self.doFP.isChecked())
		self.twoWayCheck.setEnabled(self.doFP.isChecked())
		self.threeWayCheck.setEnabled(self.doFP.isChecked())
		self.svdCheck.setEnabled(self.doFP.isChecked())
		self.pweBox.setEnabled(self.doFP.isChecked())

	def tab_two_ok_action(self):
		self.tab_two_save()
		self.close()

	def tab_two_apply_action(self):
		self.tab_two_save()

	def tab_two_restore_action(self):
		self.tab_two_init()
		pass

	def tab_two_save(self):
		self.config_dict['pweVal'] = self.pweBox.value()
		self.config_dict['doFP'] = self.doFP.isChecked()
		self.config_dict['handleMissing'] = self.handleMissingCheck.isChecked()
		self.config_dict['oneHot'] = self.oneHotCheck.isChecked()
		self.config_dict['norm'] = self.normCheck.isChecked()
		self.config_dict['logTrans'] = self.logTransCheck.isChecked()
		self.config_dict['pwe'] = self.pweCheck.isChecked()
		self.config_dict['twoWay'] = self.twoWayCheck.isChecked()
		self.config_dict['threeWay'] = self.threeWayCheck.isChecked()
		self.config_dict['svd'] = self.svdCheck.isChecked()

	#----------------------- Tab3 -----------------------#

	def tab_three_init(self):
		self.LRCheck.setChecked(True)
		self.numEvalLR.setValue(50)

		self.KNNCheck.setChecked(True)
		self.numEvalKNN.setValue(50)

		self.NBCheck.setChecked(True)
		self.numEvalNB.setValue(50)

		self.GbTreeCheck.setChecked(False)
		self.numEvalGbTree.setValue(50)

		self.GbLinearCheck.setChecked(True)
		self.numEvalGbLinear.setValue(50)

		self.kernSVMCheck.setChecked(False)
		self.numEvalKernSVM.setValue(50)

		self.LinearSVMCheck.setChecked(True)
		self.numEvalLinearSVM.setValue(50)

		self.ERTCheck.setChecked(False)
		self.numEvalERT.setValue(50)

		self.RFCheck.setChecked(False)
		self.numEvalRF.setValue(50)

		self.MLPCheck.setChecked(False)
		self.numEvalMLP.setValue(50)

		self.reg_config()

	def tab_three_ok_action(self):
		ret = self.tab_three_validate()
		if ret == True:
			self.tab_three_save()
			self.close()
		else:
			self.error_msg()

	def tab_three_apply_action(self):
		ret = self.tab_three_validate()
		if ret == False:
			self.error_msg()
		else:
			self.tab_three_save()

	def tab_three_restore_action(self):
		self.tab_three_init()
		pass

	def tab_three_save(self):
		self.config_dict['numEvalLR'] = int(self.numEvalLR.text())
		self.config_dict['numEvalKNN'] = int(self.numEvalKNN.text())
		self.config_dict['numEvalNB'] = int(self.numEvalNB.text())
		self.config_dict['numEvalGbTree'] = int(self.numEvalGbTree.text())
		self.config_dict['numEvalGbLinear'] = int(self.numEvalGbLinear.text())
		self.config_dict['numEvalPolySVM'] = int(self.numEvalKernSVM.text())
		self.config_dict['numEvalLinearSVM'] = int(self.numEvalLinearSVM.text())
		self.config_dict['numEvalERT'] = int(self.numEvalERT.text())
		self.config_dict['numEvalRF'] = int(self.numEvalRF.text())
		self.config_dict['numEvalMLP'] = int(self.numEvalMLP.text())

		self.config_dict['LR'] = self.LRCheck.isChecked()
		self.config_dict['KNN'] = self.KNNCheck.isChecked()
		self.config_dict['NB'] = self.NBCheck.isChecked()
		self.config_dict['GbTree'] = self.GbTreeCheck.isChecked()
		self.config_dict['GbLinear'] = self.GbLinearCheck.isChecked()
		self.config_dict['PolySVM'] = self.kernSVMCheck.isChecked()
		self.config_dict['LinearSVM'] = self.LinearSVMCheck.isChecked()
		self.config_dict['ERT'] = self.ERTCheck.isChecked()
		self.config_dict['RF'] = self.RFCheck.isChecked()
		self.config_dict['MLP'] = self.MLPCheck.isChecked()

	def tab_three_validate(self):
		ret = (self.LRCheck.isChecked()
			  or self.KNNCheck.isChecked()
			  or self.NBCheck.isChecked()
			  or self.GbTreeCheck.isChecked()
			  or self.GbLinearCheck.isChecked()
			  or self.kernSVMCheck.isChecked()
			  or self.LinearSVMCheck.isChecked()
			  or self.ERTCheck.isChecked()
			  or self.RFCheck.isChecked()
			  or self.MLPCheck.isChecked())
		return ret

	#----------------------- Tab4 -----------------------#

	def tab_four_init(self):

		self.numThreadCB.clear()
		self.metricCB.clear()
		self.numFoldCB.clear()

		# get number of processors
		cpuCount = multiprocessing.cpu_count()
		for i in range(1,cpuCount+1):
			self.numThreadCB.addItem(str(i))

		# assign metrics
		metrics = ['ClassificationAccuracy','AUC','LogLoss','MSE','RMSE']
		self.metricCB.addItems(metrics)

		# assign folds
		folds = [2,3,4,5,6,7,8,9,10]
		for i in range(len(folds)):
			folds[i] = str(folds[i])
		self.numFoldCB.addItems(folds)

		#set check boxes
		self.avgCheck.setChecked(True)
		self.stackCheck.setChecked(False)
		self.invCVCheck.setChecked(False)
		self.buildFinalCheck.setChecked(True)

	def tab_four_ok_action(self):
		ret = self.tab_four_save()
		if ret==True:
			self.close()

	def tab_four_apply_action(self):
		ret = self.tab_four_save()

	def tab_four_restore_action(self):
		self.tab_four_init()
		pass

	def tab_four_save(self):
		self.config_dict['avg'] = self.avgCheck.isChecked()
		self.config_dict['stack'] = self.stackCheck.isChecked()
		self.config_dict['numThread'] = int(str(self.numThreadCB.currentText()))
		self.config_dict['n_folds'] = int(str(self.numFoldCB.currentText()))
		self.config_dict['metric'] = str(self.metricCB.currentText())
		self.config_dict['inverse_kfold'] = self.invCVCheck.isChecked()
		self.config_dict['generate_final_mode'] = self.buildFinalCheck.isChecked()

		# to avoid errors -- to be removed
		self.config_dict['ensemble'] = True
		self.config_dict['verbose'] = 0
		return True

	#----------------------- Tab5 -----------------------#

	def tab_five_init(self):
		# set check boxes
		self.ROCCheck.setChecked(True)
		self.confMatCheck.setChecked(True)
		self.featImpCheck.setChecked(True)
		self.cloudCheck.setChecked(True)
		self.decBoundCheck.setChecked(True)

	def tab_five_ok_action(self):
		self.tab_five_save()
		self.close()

	def tab_five_apply_action(self):
		self.tab_five_save()

	def tab_five_restore_action(self):
		self.tab_five_init()

	def tab_five_save(self):
		self.config_dict['roc'] = self.ROCCheck.isChecked()
		self.config_dict['confMat'] = self.confMatCheck.isChecked()
		self.config_dict['featImp'] = self.featImpCheck.isChecked()
		self.config_dict['cloud'] = self.cloudCheck.isChecked()
		self.config_dict['decBound'] = self.decBoundCheck.isChecked()

	#---------------------- GENERAL ----------------------#

	def error_msg(self):
		msg = QMessageBox()
		msg.setIcon(QMessageBox.Critical)
		msg.setText("Invalid Input.")
		msg.setStandardButtons(QMessageBox.Ok)
		msg.exec_()

	def reg_config(self):
		reg = self.regressionCheck.isChecked()


		if (reg == True):
			self.KNNCheck.setChecked(not reg)
			self.NBCheck.setChecked(not reg)
			self.LinearSVMCheck.setChecked(not reg)
			self.kernSVMCheck.setChecked(not reg)
			self.ERTCheck.setChecked(not reg)

		self.KNNCheck.setEnabled(not reg)
		self.numEvalKNN.setEnabled(not reg)


		self.NBCheck.setEnabled(not reg)
		self.numEvalNB.setEnabled(not reg)


		self.LinearSVMCheck.setEnabled(not reg)
		self.numEvalLinearSVM.setEnabled(not reg)

		self.kernSVMCheck.setEnabled(not reg)
		self.numEvalKernSVM.setEnabled(not reg)

		self.ERTCheck.setEnabled(not reg)
		self.numEvalERT.setEnabled(not reg)

	def launch(self):
		self.exec_()
		return self.config_dict
