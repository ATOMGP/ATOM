from learner import *
from FeatureProcessor import *
import pandas as pd
import numpy as np
from metric import *
from utils import feature_weighted_regression_matrix, feature_weighted_classification_matrix, feature_prediction_classification_concat, feature_prediction_regression_concat


class finalModelESClassifier:
    def __init__(self, history_df, model_ind, feature_preprocessing, label_encoder):
        self.feature_preprocessing = feature_preprocessing
        self.label_encoder = label_encoder
        if isinstance(history_df['Parameters'].iloc[model_ind], basestring):
            self.architecture = eval(history_df['Parameters'].iloc[model_ind])
        else:
            self.architecture = history_df['Parameters'].iloc[model_ind]
        self.learners = {}
        for bag in self.architecture:
            for k, v in bag.items():
                if isinstance(history_df['Parameters'].iloc[k], basestring):
                    params = eval(history_df['Parameters'].iloc[k])
                else:
                    params = history_df['Parameters'].iloc[k]
                if k not in self.learners:                
                    self.learners[k] = eval(history_df['Learner'].iloc[k])(**params)
    
    def fit(self, X, Y):
        for learner in self.learners.values():
            learner.fit(X, Y)
    
    def predict_proba(self, X):
        X_test = self.feature_preprocessing.transform(X)
        predictions = None
        n_bags = 0
        for bag in self.architecture:
            n_bags += 1
            p = None
            sum_w = 0
            for k, w in bag.items():
                if p is None:
                    p = w * self.learners[k].predict_proba(X_test)
                else:
                    p += w * self.learners[k].predict_proba(X_test)
                sum_w += w
            p /= sum_w
            if predictions is None:
                predictions = p
            else:
                predictions += p
        predictions /= n_bags
        return predictions
        
    def predict(self, X):
        predictions = self.predict_proba(X)
        p_class = np.argmax(predictions, axis = 1)
        return self.label_encoder.inverse_transform(p_class)
        
class finalModelStackingClassifier:
    def __init__(self, history_df, best_df, model_ind, feature_preprocessing, label_encoder):
        self.best_df = best_df
        self.feature_preprocessing = feature_preprocessing
        self.learners = []
        self.label_encoder = label_encoder
        for i in range(best_df.shape[0]):
            if isinstance(best_df['Parameters'].iloc[i], basestring):
                params = eval(best_df['Parameters'].iloc[i])
            else:
                params = best_df['Parameters'].iloc[i]
                
            learner = eval(best_df['Learner'].iloc[i])(**params)                                
            self.learners.append(learner)
        
        if history_df['Learner'].iloc[model_ind] == 'Linear_Stacking':
            self.stacker = LogisticRegression()
        else:
            xgb_param = {'max_depth':13, 'eta':0.01, 'subsample':0.75, 'colsample_bytree':0.68, 'min_child_weight':1}
            self.stacker = XGBTreeClassifier(**xgb_param)
            
        if isinstance(history_df['Parameters'].iloc[model_ind], basestring):
            stack_params = eval(history_df['Parameters'].iloc[model_ind])
        else:
            stack_params = history_df['Parameters'].iloc[model_ind]
        self.features = stack_params['Features']
        self.feature_weighted = stack_params['Feature Weighted']

    def _load_predictions(self, X, Y):
        X_p = np.zeros((self.best_df.shape[0], X.shape[0], np.unique(Y).shape[0]))
        i = 0
        for path in self.best_df['Path'].values:
            matrix = np.load(path)
            X_p[i] = matrix['PREDICTIONS']
            i += 1
        return X_p
        
    def fit(self, X, Y):
        for learner in self.learners:
            learner.fit(X, Y)
        X_p = self._load_predictions(X, Y)
        if self.feature_weighted:
            X_train = feature_weighted_classification_matrix(X_p, X)
        else:
            X_train = feature_prediction_classification_concat(X_p, X)
        self.stacker.fit(X_train, Y)
        
    def predict_proba(self, X):
        X_test = self.feature_preprocessing.transform(X)
        X_p_test = np.zeros((len(self.learners), X_test.shape[0], self.label_encoder.classes_.shape[0]))
        i = 0
        for learner in self.learners:
            X_p_test[i] = learner.predict_proba(X_test)
            i += 1
        print X_p_test.shape, X.shape
        if self.feature_weighted:
            X_test = feature_weighted_classification_matrix(X_p_test, X_test)
        else:
            X_test = feature_prediction_classification_concat(X_p_test, X_test)   
        return self.stacker.predict_proba(X_test)     
        
    def predict(self, X):
        predictions = self.predict_proba(X)
        p_class = np.argmax(predictions, axis = 1)
        return self.label_encoder.inverse_transform(p_class)        
        
class finalModelESRegressor:
    def __init__(self, history_df, model_ind, feature_preprocessing):
        self.feature_preprocessing = feature_preprocessing
        if isinstance(history_df['Parameters'].iloc[model_ind], basestring):
            self.architecture = eval(history_df['Parameters'].iloc[model_ind])
        else:
            self.architecture = history_df['Parameters'].iloc[model_ind]
        self.learners = {}
        for bag in self.architecture:
            for k, v in bag.items():
                if isinstance(history_df['Parameters'].iloc[k], basestring):
                    params = eval(history_df['Parameters'].iloc[k])
                else:
                    params = history_df['Parameters'].iloc[k]
                if k not in self.learners:                
                    self.learners[k] = eval(history_df['Learner'].iloc[k])(**params)
    
    def fit(self, X, Y):
        for learner in self.learners.values():
            learner.fit(X, Y)
    
    def predict_proba(self, X):
        X_test = self.feature_preprocessing.transform(X)
        ppredictions = None
        n_bags = 0
        for bag in self.architecture:
            n_bags += 1
            p = None
            sum_w = 0
            for k, w in bag.items():
                if p is None:
                    p = w * self.learners[k].predict(X_test)
                else:
                    p += w * self.learners[k].predict(X_test)
                sum_w += w
            p /= sum_w
            if predictions is None:
                predictions = p
            else:
                predictions += p
        predictions /= n_bags
        return predictions
        
    def predict(self, X):
        predictions = self.predict_proba(X)
        return predictions
        
class finalModelStackingRegressor:
    def __init__(self, history_df, best_df, model_ind, feature_preprocessing):
        self.best_df = best_df
        self.feature_preprocessing = feature_preprocessing
        self.learners = []
        for i in range(best_df.shape[0]):
            if isinstance(best_df['Parameters'].iloc[i], basestring):
                params = eval(best_df['Parameters'].iloc[i])
            else:
                params = best_df['Parameters'].iloc[i]
                
            learner = eval(best_df['Learner'].iloc[i])(**params)                                
            self.learners.append(learner)
        
        if history_df['Learner'].iloc[model_ind] == 'Linear_Stacking':
            self.stacker = RidgeRegression()
        else:
            xgb_param = {'max_depth':13, 'eta':0.01, 'subsample':0.75, 'colsample_bytree':0.68, 'min_child_weight':1}
            self.stacker = XGBTreeRegressor(**xgb_param)
            
        if isinstance(history_df['Parameters'].iloc[model_ind], basestring):
            stack_params = eval(history_df['Parameters'].iloc[model_ind])
        else:
            stack_params = history_df['Parameters'].iloc[model_ind]
            
        self.features = stack_params['Features']
        self.feature_weighted = stack_params['Feature Weighted']

    def _load_predictions(self, X, Y):
        self.X_p = np.zeros((best_df.shape[0], self.X.shape[0]))
        i = 0
        for path in self.best_df['Path'].values:
            matrix = np.load(path)
            X_p[i] = matrix['PREDICTIONS']
            i += 1
        return X_p
        
    def fit(self, X, Y):
        for learner in self.learners:
            learner.fit(X, Y)
        X_p = self._load_predictions(X, Y)
        if self.feature_weighted:
            X_train = feature_weighted_regression_matrix(X_p, X)
        else:
            X_train = feature_prediction_regression_concat(X_p, X)
        self.stacker.fit(X_train, Y)
        
    def predict_proba(self, X):
        X_test = self.feature_preprocessing.transform(X)
        X_p_test = np.zeros((len(self.learners), X_test.shape[0], 1))
        i = 0
        for learner in self.learners:
            X_p_test[i] = learner.predict_proba(X_test)
            i += 1
        if self.feature_weighted:
            X_test = feature_weighted_regression_matrix(X_p_test, X_test)
        else:
            X_test = feature_prediction_regression_concat(X_p_test, X_test)   
        return self.stacker.predict(X_test)     
        
    def predict(self, X):
        predictions = self.predict_proba(X)
        return predictions
