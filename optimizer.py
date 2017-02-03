import numpy as np
import pandas as pd
from time import time, sleep
import cPickle as pickle
from hyperopt import fmin, tpe, hp, Trials
import os
from sklearn.externals.joblib import Parallel, delayed
from multiprocessing import cpu_count
from itertools import product
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import ParameterSampler
# from scipy.stats.distributions import 
from abc import ABCMeta, abstractmethod
from shared import *
    
def cv(learner, X, Y, train, test):
    learner.fit(X[train], Y[train])
    return learner.predict_proba(X[test])
    
class BaseOptimizer(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self, X, y, kf, learner, metric, maximize=True, overwrite=False, path = '', verbose=0):
        self.X = X
        self.y = y
        self.kf = kf
        self.learner = learner
        self.metric = metric
        self.maximize = maximize
        self.path = path
        self.verbose = verbose
        #exmple
        #for i in range (101):
        #    shared_mem.write('learner#'+str(i), i)
        #    time.sleep(1)
        if (not (os.path.isfile(self.path + self.learner.__name__ + '.csv') and os.path.isfile(self.path + 'trials_' + self.learner.__name__))) or overwrite == True:
            # print self.path + self.learner.__name__ + '.csv', self.path + 'trials_' + self.learner.__name__, overwrite
            self.results = pd.DataFrame(columns = ["Learner", "Score Mean", "Score Std", "Parameters", "Path"])
            self.trials = Trials()
            self.iter_val = 0
        else:
            self.load_state()
            
    def get_avg_score(self,predictions):
        '''
        get average score for the folds
        '''
        scores = []
        for train, test in self.kf:
            scores.append(self.metric.evaluate(self.y[test], predictions[test]))
        return (np.mean(scores), np.std(scores))

    def append_results(self, indx, mean, std, param):
        '''
        append learner name, score, used parameters to the dataframe
        '''
        self.results.loc[indx] = [self.learner.__name__, mean, std, param, self.gen_path(self.path)+'.npz']
        
    def save_state(self):
        self.results.to_csv(self.path + self.learner.__name__ + '.csv', index=False)
        with open(self.path + 'trials_' + self.learner.__name__, 'wb') as f:
            pickle.dump(self.trials, f)
     
    def load_state(self):
        self.results = pd.read_csv(self.path + self.learner.__name__ + '.csv')
        with open(self.path + 'trials_' + self.learner.__name__, 'rb') as f:
            self.trials = pickle.load(f)
        self.iter_val = len(self.results)
        
    def get_results(self):
        return self.results

    def score(self, param):
        '''
        get score from the learner cross validation function, increment iteration,
        append to dataframe
        '''
        try:
            t = time()
            for k, v in self.learner.space.iteritems():
                if v[3] == 'discrete':
                    param[k] = int(param[k])
            if self.learner.parallel_cv == False:
                learner = self.learner(**param)
                predictions = learner.cv(self.X, self.y, self.kf)
            else:
                preds = Parallel(n_jobs=min(cpu_count(), len(self.kf)), verbose=self.verbose)(delayed(cv)(self.learner(**param), self.X, self.y, train, test) for train, test in self.kf)
                predictions = np.zeros((self.y.shape[0], len(np.unique(self.y))))
                i = 0
                for train, test in self.kf:
                    predictions[test] = preds[i]
                    i += 1
            self.save_predictions(predictions)
            current_mean, current_std = self.get_avg_score(predictions)
            if self.verbose > 0:
                print self.iter_val, param
                print "score mean:", current_mean, "score std:", current_std, "time:", time() - t
            self.append_results(self.iter_val, current_mean, current_std, param)
            if self.maximize == True:
                return -current_mean
            else:
                return current_mean
        except Exception as e:
            if self.maximize == True:
                current_mean,current_std = 10**9, 0
            else:
                current_mean,current_std = -10**9, 0 
            if self.verbose > 0:
                print "Exception:", e
                print self.iter_val, param
                print "score mean:", current_mean, "score std:", current_std, "time:", time() - t
            # self.append_results(self.iter_val, current_mean, current_std, param)
            return current_mean

    def gen_path(self, path):
        return (path + self.learner.__name__ + "_" + str(self.iter_val))

    def save_predictions(self, predictions):
        np.savez(self.gen_path(self.path), PREDICTIONS = predictions)

        
class GridSearch(BaseOptimizer):
    def __init__(self, X, y, kf, learner, metric, maximize=True, overwrite=False, path = '', verbose=0):
        super(GridSearch, self).__init__(X, y, kf, learner, metric, maximize, overwrite, path, verbose)
        
    def generate_grid(self, learner_grid):
        # values = list(product(*learner_grid.values()))
        # grid = [{learner_grid.keys()[i]: y[i] for i in range(len(y))} for y in values]
        return list(ParameterGrid(learner_grid))
        
    def optimize(self, shared_mem, past_evals, total_evals, max_evals=100):
        '''
        learner: class inistantiator e.g. DecisionTreeClassifier not DecisionTreeClassifier()
        '''
        grid = self.generate_grid(self.learner.grid)
        bst = None
        m = -np.inf if self.maximize else np.inf
        for i in range(self.iter_val, len(grid)):
            param = grid[i]
            sc = self.score(param)
            if self.maximize == True:
                # print '1', sc, m
                bst = param if -sc > m else bst
                m = -sc if -sc > m else m
            else:
                # print '2', -sc < m
                bst = param if sc < m else bst
                m = sc if sc < m else m
            if not (shared_mem is None):
                msg = "Classifier: "+ self.learner.__name__ + " -- " + "Score: " + str(sc)
                tmp = Shared()
                tmp.write(msg, (float(past_evals + self.iter_val)/float(total_evals))*100)
                shared_mem.put(tmp)
            self.save_state()
            self.iter_val += 1
        return bst
        

class HyperoptOptimizer(BaseOptimizer):
    def __init__(self, X, y, kf, learner, metric, maximize=True, overwrite=False, path = '', verbose=0):
        super(HyperoptOptimizer, self).__init__(X, y, kf, learner, metric, maximize, overwrite, path, verbose)
        
    def parse_search_space(self, learner_space):
        '''
        search space is dictionary
        {'n_estimators': ('uniform', 1, 1000, 'discrete')}
        '''
        search_space = dict()
        for k, v in learner_space.iteritems():
            if v[2] == 'samples':
                v = (v[0], v[1], min(100, self.X.shape[0]/len(self.kf)-1), v[3])
            if v[3] == 'discrete':
                search_space[k] = hp.quniform(k, v[1], v[2], 1)
            elif v[0] == 'uniform':
                search_space[k] = hp.uniform(k, v[1], v[2])
            elif v[0] == 'loguniform':
                search_space[k] = hp.loguniform(k, v[1], v[2])
            elif v[0] == 'normal':
                search_space[k] = hp.normal(k, v[1], v[2])
            elif v[0] == 'lognormal':
                search_space[k] = hp.lognormal(k, v[1], v[2])
            elif v[0] == 'quniform':
                search_space[k] = hp.quniform(k, v[1], v[2], v[3])
            elif v[0] == 'qloguniform':
                search_space[k] = hp.qloguniform(k, v[1], v[2], v[3])
            elif v[0] == 'qnormal':
                search_space[k] = hp.qnormal(k, v[1], v[2], v[3])
            elif v[0] == 'qlognormal':
                search_space[k] = hp.qlognormal(k, v[1], v[2], v[3])
        return search_space

    def optimize(self, shared_mem, past_evals, total_evals, max_evals=100):
        '''
        learner: class inistantiator e.g. DecisionTreeClassifier not DecisionTreeClassifier()
        '''
        bst = None
        for i in range(self.iter_val+1, max_evals+1):
            bst = fmin(self.score, space=self.parse_search_space(self.learner.space), algo=tpe.suggest, trials=self.trials, max_evals=i)
            
            if not (shared_mem is None) and self.results.shape[0]-1>0:
                sc = float(self.results['Score Mean'].iloc[self.results.shape[0]-1])
                msg = "Classifier: "+ self.learner.__name__ + " -- " + "Score: " + str(sc)
                tmp = Shared()
                tmp.write(msg, (float(past_evals + self.iter_val)/float(total_evals))*100)
                shared_mem.put(tmp)            
                self.save_state()
            self.iter_val += 1
        return bst

