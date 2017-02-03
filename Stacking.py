import numpy as np
import pandas as pd
from learner import LogisticRegression, RidgeRegression, XGBTreeClassifier, XGBTreeRegressor
from utils import feature_weighted_regression_matrix, feature_weighted_classification_matrix, feature_prediction_classification_concat, feature_prediction_regression_concat

class Stacking():
    def __init__(self, X, Y, best_df, regression, kf, path, metric):
        self.results = pd.DataFrame(columns = ["Learner", "Score Mean", "Score Std", "Parameters", "Path"])
        self.X = X
        self.Y = Y
        self.kf = kf
        self.path = path
        self.metric = metric
        self.regression = regression
        xgb_param = {'max_depth':13, 'eta':0.01, 'subsample':0.75, 'colsample_bytree':0.68, 'min_child_weight':1}
        if regression:
            self.stackers = [RidgeRegression(), XGBTreeRegressor(**xgb_param)]
        else:
            self.stackers = [LogisticRegression(), XGBTreeClassifier(**xgb_param)]
        self._load_predictions(best_df, regression)
        
    def _load_predictions(self,best_df, regression):
        if not regression:
            self.X_p = np.zeros((best_df.shape[0], self.X.shape[0], np.unique(self.Y).shape[0]))
        else:
            self.X_p = np.zeros((best_df.shape[0], self.X.shape[0]))
        i = 0
        for path in best_df['Path'].values:
            matrix = np.load(path)
            print self.X_p.shape, matrix['PREDICTIONS'].shape
            self.X_p[i] = matrix['PREDICTIONS']
            i += 1

    def append_results(self, indx, learner, mean, std, param):
        self.results.loc[indx] = [learner, mean, std, param, self.path + learner + '.npz']

    def save_predictions(self, predictions, learner):
        np.savez(self.path + learner, PREDICTIONS = predictions)
            
    def _stack_feature(self, weighted):
        if self.regression:
            if weighted:
                return feature_weighted_regression_matrix(self.X_p, self.X)
            return feature_prediction_regression_concat(self.X_p, self.X)        
        else:
            if weighted:
                return feature_weighted_classification_matrix(self.X_p, self.X)
            return feature_prediction_classification_concat(self.X_p, self.X)
    
    def get_results(self):
        return self.results
        
    def _scorer(self, p):
        sc = []
        for _, test in self.kf:
            sc.append(self.metric.evaluate(self.Y[test], p[test]))
        return np.mean(sc), np.std(sc)
    
    def run(self):
        X_train = self._stack_feature(False)
        p = self.stackers[0].cv(X_train, self.Y, self.kf)
        mean, std = self._scorer(p)
        self.append_results(0, 'Linear_Stacking', mean, std, {'Features': True,'Feature Weighted':False})
        self.save_predictions(p, 'Linear_Stacking')
        p = self.stackers[1].cv(X_train, self.Y, self.kf)
        mean, std = self._scorer(p)
        self.append_results(1, 'Non_Linear_Stacking', mean, std, {'Features': True,'Feature Weighted':False})
        self.save_predictions(p, 'Non_Linear_Stacking')
        
        X_train = self._stack_feature(True)
        p = self.stackers[0].cv(X_train, self.Y, self.kf)
        mean, std = self._scorer(p)
        self.append_results(2, 'Linear_Stacking', mean, std, {'Features': True,'Feature Weighted':True})
        self.save_predictions(p, 'Linear_Stacking')
        p = self.stackers[1].cv(X_train, self.Y, self.kf)
        mean, std = self._scorer(p)
        self.append_results(3, 'Non_Linear_Stacking', mean, std, {'Features': True,'Feature Weighted':True})
        self.save_predictions(p, 'Non_Linear_Stacking')
