from learner import *
from optimizer import HyperoptOptimizer
import pandas as pd
import numpy as np
from kfold import *
from FeatureProcessor import FeatureProcessor
from metric import *
from ensembleSelection import *
from sklearn.preprocessing import LabelEncoder
import os
import time
from finalModel import finalModelESClassifier, finalModelStackingClassifier, finalModelESRegressor, finalModelStackingRegressor
from shared import Shared
from utils import *
from report_generator import *
from Stacking import Stacking
from shared import *

class Manager:
    def __init__(self, pref_dict):
        self._initialize_variables()
        self.set_pref(pref_dict)
        self.read_df()
        self._set_target_vector()
    
    def _initialize_variables(self):
        self.project_dir = None
        self.project_name = None
        self.path = None
        
        self.one_dataset = None
        self.inverse_kfold = None 
        self.n_folds = None 
        self.train_data_path = None
        self.test_data_path = None
        self.target_variable_name = None
        self.regression = None

        self.pair_wise_elimination = None
        self.two_way_features = None
        self.three_way_features = None
        self.svd_features = None

        self.learners = []
        self.n_evals = {}
        self.total_evals = 0
        self.metric = None
        self.verbose = None

        self.run_ensemble = None 
        self.generate_final_model = None 

        self.n_threads = None

        self.train_df = None
        self.X = None
        self.test_df = None
        self.X_test = None

        self.Y = None
        self.n_classes = None
        self.kf = None
        self.generate_test_prediction = False
        
        self.history_df = None
        self.best_df = pd.DataFrame(columns = ["Learner", "Score Mean", "Score Std", "Parameters", "Path"])

        self.X_p = None

        self.label_encoder = LabelEncoder()
        self.feature_processor = FeatureProcessor()
        self.ensemble_selection = None
        self.stacking = None

    def set_pref(self, pref_dict):
        #project information
        self.project_dir = pref_dict['projectLoc']
        self.project_name = pref_dict['projectName']
        if not os.path.isdir(self.project_dir + '/' +self.project_name):
            os.mkdir(self.project_dir + '/' +self.project_name)
        
        self.path = self.project_dir +  '/' + self.project_name +  '/'
        
        self.one_dataset = pref_dict['oneDataset']
        self.inverse_kfold = pref_dict['inverse_kfold'] 
        self.n_folds = pref_dict['n_folds'] 
        self.train_data_path = pref_dict['trainPath']
        self.test_data_path = pref_dict['testPath']
        self.target_variable_name = pref_dict['target']
        self.regression = pref_dict['regression']

        self.pair_wise_elimination = pref_dict['pwe']
        self.two_way_features = pref_dict['twoWay']
        self.three_way_features = pref_dict['threeWay']
        self.svd_features = pref_dict['svd']

        if 'LR' in pref_dict.keys() and pref_dict['LR']:
            if not self.regression:
                self.learners.append(LogisticRegression)
                self.n_evals[LogisticRegression.__name__] = pref_dict['numEvalLR']
            else:
                self.learners.append(LassoRegression)
                self.n_evals[LassoRegression.__name__] = pref_dict['numEvalLR']
                self.learners.append(RidgeRegression)
                self.n_evals[RidgeRegression.__name__] = pref_dict['numEvalLR']  
            self.total_evals += pref_dict['numEvalLR']              

        if 'KNN' in pref_dict.keys() and pref_dict['KNN']:
            if not self.regression:
                self.learners.append(KNeighborsClassifier)
                self.n_evals[KNeighborsClassifier.__name__] = pref_dict['numEvalKNN']
            else:
                self.learners.append(KNeighborsRegressor)
                self.n_evals[KNeighborsRegressor.__name__] = pref_dict['numEvalKNN']  
            self.total_evals += pref_dict['numEvalKNN']              

        if 'NB' in pref_dict.keys() and pref_dict['NB']:
            if not self.regression:
                #self.learners.append(MultinomialNB)
                #self.n_evals[MultinomialNB.__name__] = pref_dict['numEvalNB']
                self.learners.append(BernoulliNB)
                self.n_evals[BernoulliNB.__name__] = pref_dict['numEvalNB']
                self.total_evals += pref_dict['numEvalNB']

        if 'GbTree' in pref_dict.keys() and pref_dict['GbTree']:
            if not self.regression:
                self.learners.append(XGBTreeClassifier)
                self.n_evals[XGBTreeClassifier.__name__] = pref_dict['numEvalGbTree']
            else:
                self.learners.append(XGBTreeRegressor)
                self.n_evals[XGBTreeRegressor.__name__] = pref_dict['numEvalGbTree']   
            self.total_evals += pref_dict['numEvalGbTree']         

        if 'GbLinear' in pref_dict.keys() and pref_dict['GbLinear']:
            if not self.regression:
                self.learners.append(XGBLinearClassifier)
                self.n_evals[XGBLinearClassifier.__name__] = pref_dict['numEvalGbLinear']
            else:
                self.learners.append(XGBLinearRegressor)
                self.n_evals[XGBLinearRegressor.__name__] = pref_dict['numEvalGbLinear']
            self.total_evals += pref_dict['numEvalGbLinear']              

        if 'RBFSVM' in pref_dict.keys() and pref_dict['RBFSVM']: 
            if not self.regression:
                self.learners.append(RBF_SVC)
                self.n_evals[RBF_SVC.__name__] = pref_dict['numEvalRBFSVM']
            else:
                self.learners.append(RBF_SVR)
                self.n_evals[RBF_SVR.__name__] = pref_dict['numEvalRBFSVM']
            self.total_evals += pref_dict['numEvalRBFSVM']  
                
        if 'PolySVM' in pref_dict.keys() and pref_dict['PolySVM']: 
            if not self.regression:
                self.learners.append(Poly_SVC)
                self.n_evals[Poly_SVC.__name__] = pref_dict['numEvalPolySVM']
            else:
                self.learners.append(Poly_SVR)
                self.n_evals[Poly_SVR.__name__] = pref_dict['numEvalPolySVM']
            self.total_evals += pref_dict['numEvalPolySVM']  
                
        if 'LinearSVM' in pref_dict.keys() and pref_dict['LinearSVM']:
            if not self.regression:
                self.learners.append(Linear_SVC)
                self.n_evals[Linear_SVC.__name__] = pref_dict['numEvalLinearSVM']
            else:
                self.learners.append(Linear_SVR)
                self.n_evals[Linear_SVR.__name__] = pref_dict['numEvalLinearSVM']
            self.total_evals += pref_dict['numEvalLinearSVM']  
    
        if 'ERT' in pref_dict.keys() and pref_dict['ERT']:
            if not self.regression:
                self.learners.append(ExtraTreesClassifier)
                self.n_evals[ExtraTreesClassifier.__name__] = pref_dict['numEvalERT']
            else:
                self.learners.append(ExtraTreesRegressor)
                self.n_evals[ExtraTreesRegressor.__name__] = pref_dict['numEvalERT']
            self.total_evals += pref_dict['numEvalERT']              
            
        if 'RF' in pref_dict.keys() and pref_dict['RF']:
            if not self.regression:
                self.learners.append(RandomForestClassifier)
                self.n_evals[RandomForestClassifier.__name__] = pref_dict['numEvalRF']
            else:
                self.learners.append(RandomForestRegressor)
                self.n_evals[RandomForestRegressor.__name__] = pref_dict['numEvalRF']
            self.total_evals += pref_dict['numEvalRF']                 

        if pref_dict['MLP']:
           self.learners.append(NeuralNetworkClassifier)
           self.n_evals[NeuralNetworkClassifier.__name__] = pref_dict['numEvalMLP']
        #else:
        #    self.learners.append(MLPRegressor)
        #    self.n_evals[MLPRegressor.__name__] = pref_dict['numEvalMLP']        

        self.run_ensemble = pref_dict['ensemble'] 
        self.generate_final_model = pref_dict['generate_final_mode'] 

        self.n_threads = pref_dict['numThread']

        self.metric = eval(pref_dict['metric'])()
            
        self.verbose = pref_dict['verbose']

    def _set_target_vector(self):
        if not self.regression:
            self.Y = self.label_encoder.fit_transform(self.train_df[self.target_variable_name].values).flatten()
            self.n_classes = np.unique(self.Y).shape[0] 
        else:
            self.Y = self.train_df[self.target_variable_name].values.flatten()
        self.train_df.drop([self.target_variable_name], axis=1, inplace=True)
        if self.inverse_kfold:
            self.kf = InverseKFold(self.n_folds, self.Y)
        else:
            self.kf = StratifiedKFold(self.Y, self.n_folds)
    
    def read_df(self):
        if self.one_dataset:
            try:
                self.train_df = pd.read_csv(self.train_data_path)
                return (True, self.train_df.keys())
            except:
                return (False, 'wrong_input_file_format')
        else:
            try:
                self.train_df = pd.read_csv(self.train_data_path)
                self.test_df = pd.read_csv(self.test_data_path)
                self.generate_test_prediction = True
                return (True, self.train_df.keys(), self.test_df.keys())
            except:
                return (False, 'wrong_input_file_format')
    
    def load_prediction_matrix(self, level):
        if level == 0:
            self.X = self.feature_processor.fit_transform(self.train_df, True, self.three_way_features, self.two_way_features, self.pair_wise_elimination, self.svd_features, self.Y)
        else:
            if 'level' in self.history_df.keys():
                level_df = self.history_df.loc[history_df['level'] == level]
            else:
                level_df = self.history_df
        
            if not self.regression:
                self.X_p = np.zeros((level_df.shape[0], self.train_df.shape[0], self.n_classes))
            else:
                self.X_p = np.zeros((level_df.shape[0], self.train_df.shape[0]))
            i = 0
            for path in level_df['Path'].values:
                matrix = np.load(path)
                self.X_p[i] = matrix['PREDICTIONS']
                i += 1
            
    def build_library(self, level, shared_mem):
        past_evals = 0
        for learner in self.learners:
            opt = HyperoptOptimizer(self.X, self.Y, self.kf, learner, self.metric, path = self.path, maximize = self.metric.maximize, verbose=self.verbose)
            bst = opt.optimize(shared_mem, past_evals, self.total_evals, self.n_evals[learner.__name__])
            past_evals += self.n_evals[learner.__name__]
            df = opt.get_results()
            if self.history_df is None:
                self.history_df = df
            else:
                self.history_df = pd.concat([self.history_df, df], axis = 0, ignore_index = True)
                
            if self.metric.maximize:
                best_ind = df["Score Mean"].argmax()
            else:
                best_ind = df["Score Mean"].argmin()  
            #print df.loc[best_ind]        
            it = self.best_df.shape[0] 
            self.best_df.loc[it] = df.loc[best_ind]

    #to be implemented 
    def check_memory(self):
        return True

    def ensemble(self, level):
        if self.ensemble_selection is None:
            self.ensemble_selection = ensembleSelection(self.metric, self.kf, self.path)
        if self.check_memory():
            self.load_prediction_matrix(level)
        self.ensemble_selection.es_with_bagging(self.X_p, self.Y)
        df = self.ensemble_selection.get_results()
        self.history_df = pd.concat([self.history_df, df], axis = 0, ignore_index = True)
        
        if self.stacking is None:
            self.stacking = Stacking(self.X, self.Y, self.best_df, self.regression, self.kf, self.path, self.metric)
        self.stacking.run()
        df = self.stacking.get_results()
        self.history_df = pd.concat([self.history_df, df], axis = 0, ignore_index = True)
        
    def final_model(self):
        if self.metric.maximize:
            best_ind = self.history_df["Score Mean"].argmax()
        else:
            best_ind = self.history_df["Score Mean"].argmin()
            
        if self.regression:
            if self.history_df["Learner"].iloc[best_ind] == 'ensembleSelection':
                self.final_model = finalModelESRegressor(self.history_df, best_ind, self.feature_processor)
            elif 'Stacking' in self.history_df["Learner"].iloc[best_ind]:
                self.final_model = finalModelStackingRegressor(self.history_df, self.best_df, best_ind, self.feature_processor)
            else:
                self.final_model = finalModelESRegressor(self.history_df, -2, self.feature_processor)
        else:
            if self.history_df["Learner"].iloc[best_ind] == 'ensembleSelection':
                self.final_model = finalModelESClassifier(self.history_df, best_ind, self.feature_processor, self.label_encoder)
            elif 'Stacking' in self.history_df["Learner"].iloc[best_ind]:
                self.final_model = finalModelStackingClassifier(self.history_df, self.best_df, best_ind, self.feature_processor,  self.label_encoder)
            else:
                self.final_model = finalModelESClassifier(self.history_df, -2, self.feature_processor, self.label_encoder)
        
        self.final_model.fit(self.X, self.Y)
        print 'final_model finished'
        save(self.path + 'final_model_api' , self.final_model)
        
    def generate_report(self):
        if self.regression == True:
            classes = None
        else:
            classes = self.label_encoder.classes_
        self.report = ReportGenerator(X_train = self.X, y_train = self.Y,
                           target_names = classes,
                           path = self.path,
                           experiments = self.history_df, num_of_iter= self.total_evals,
                           maximize = self.metric.maximize, regression = self.regression, elapsed_time= 0,
                           final_model_time = 0)
        self.report.generate()
    
    def run(self, shared_mem = None):    
        self.load_prediction_matrix(0)
        self.build_library(0, shared_mem)
        if self.run_ensemble:
            self.ensemble(1)
        self.history_df.to_csv(self.path + 'hist.csv')
        tmp=Shared()
        #if self.generate_final_model:
            #tmp.write('generating final model', -1)
            #shared_mem.put(tmp)
            #self.final_model()
            #tmp.write('generated final model', -1)
            #shared_mem.put(tmp)
        
        tmp.write('generated final report', -1)
        shared_mem.put(tmp)
        self.generate_report()
        
