from sklearn.metrics import *
import numpy as np

class RMSE:
    def __init__(self):
        self.maximize = False
        self.only_binary = False
        
    def evaluate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)**0.5
        
class MSE:
    def __init__(self):
        self.maximize = False
        self.only_binary = False
        
    def evaluate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

class LogLoss:
	def __init__(self):
		self.maximize = False
		self.only_binary = False
		
	def evaluate(self, y_true, y_pred):
		return log_loss(y_true, y_pred)
		
class AUC:
    def __init__(self):
        self.maximize = True
        self.only_binary = False

    def evaluate(self, y_true, y_pred):
        # y_true_factorized = np.zeros((y_true.shape[0], np.unique(y_true).shape[0]))
        # for i in range(y_true.shape[0]):
            # y_true_factorized[i,y_true[i]] = 1
        if len(y_pred.shape) > 1:
            return roc_auc_score(y_true, y_pred[:,1])
        else:
            return roc_auc_score(y_true, y_pred)
		
class ClassificationAccuracy:
	def __init__(self):
		self.maximize = True
		self.only_binary = False
		
	def evaluate(self, y_true, y_pred):
		y_pred_class = np.argmax(y_pred, axis=1)
		
		return accuracy_score(y_true, y_pred_class)
