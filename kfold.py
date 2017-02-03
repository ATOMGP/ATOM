from sklearn.cross_validation import StratifiedKFold

class KFold():
	def __init__(self, k, Y):
		self.kf = StratifiedKFold(Y, n_folds = k, shuffle = True)
		
	def __iter__(self):
		for train_ind, test_ind in self.kf:
			yield (train_ind, test_ind)
			
	def __len__(self):
		return len(self.kf)
			
class InverseKFold(StratifiedKFold):
	def __init__(self, k, Y):
		self.kf = StratifiedKFold(Y, n_folds = k, shuffle = True)
		
	def __iter__(self):
		for test_ind, train_ind in self.kf:
			yield (train_ind, test_ind)
			
	def __len__(self):
		return len(self.kf)
