import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

class ensembleSelection:

    def __init__(self, metric, kf, path):
        self.metric = metric
        self.kf = kf
        self.path = path
        self.results = pd.DataFrame(columns = ["Learner", "Score Mean", "Score Std", "Parameters", "Path"])
        
    def _compare(self, sc1, sc2):
        if self.metric.maximize:
            if sc1 > sc2:
                return True
            return False
        else:
            if sc1 < sc2:
                return True
            return False            
        
    def _scorer(self, y, p):
        sc = []
        for _, test_ind in self.kf:
            sc.append(self.metric.evaluate(y[test_ind], p[test_ind]))
        return np.mean(sc), np.std(sc)
        
    def _initialize(self, X_p, y):
        current_sc, current_sc_std = self._scorer(y, X_p[0])
        ind = 0
        for i in range(1, X_p.shape[0]):
            sc, sc_std = self._scorer(y, X_p[i])
            if self._compare(sc, current_sc):
                current_sc = sc
                current_sc_std = sc_std
                ind = i
        return (ind, current_sc, current_sc_std)
        
    def get_results(self):
        return self.results

    def gen_path(self, name):
        return (self.path + name)
        
    def save_predictions(self, predictions, name):
        np.savez(self.gen_path(name), PREDICTIONS = predictions)
        
    def es_with_replacement(self, X_p, y):
        model_weight = [0 for i in range(X_p.shape[0])]
        best_ind, best_sc, best_sc_std = self._initialize(X_p, y)
        current_sc = best_sc
        current_sc_std = best_sc_std
        sumP = np.copy(X_p[best_ind])
        model_weight[best_ind] += 1
        i = 1
        while True:
            i += 1
            ind = -1
            for m in range(X_p.shape[0]):
                sc, sc_std = self._scorer(y, (sumP+X_p[m])/i)
                if self._compare(sc, current_sc):
                    current_sc = sc
                    current_sc_std = sc_std
                    ind = m
            if ind>-1:
                sumP += X_p[ind]
                model_weight[ind] += 1
            else:
                break
        sumP /= (i-1)
        final_model = {i:model_weight[i] for i in range(X_p.shape[0]) if model_weight[i]>0}
        
        return (current_sc, current_sc_std, model_weight, sumP)
        
    def es_with_bagging(self, X_p, y, f = 0.5, n_bags = 20):
        list_of_indecies = [i for i in range(X_p.shape[0])]
        rs = check_random_state(4321)
        bag_size = int(f*X_p.shape[0])
        sumP = None
        final_model = []
        for i in range(n_bags):
            model_weight = [0 for j in range(X_p.shape[0])]
            rng = rs.permutation(list_of_indecies)[:bag_size]
            sc, sc_std, part_model, p = self.es_with_replacement(X_p[rng], y)
            if sumP is None:
                sumP = p
            else:
                sumP += p
            k = 0
            model = {}
            for j in rng:
                model_weight[j] = part_model[k]
                if model_weight[j]>0:
                    model[j] = model_weight[j]
                k += 1
            final_model.append(model)
            #self.save_predictions(p, 'es_with_replacement_' + str(i))
            #self.results.loc[i] = ['es_with_replacement', sc, sc_std, model, self.gen_path('es_with_replacement_' + str(i))]
                
        sumP /= n_bags
        final_sc, final_sc_std = self._scorer(y, sumP)
        self.save_predictions(sumP, 'ensembleSelection')
        self.results.loc[n_bags] = ['ensembleSelection', final_sc, final_sc_std, final_model, self.gen_path('ensembleSelection') + '.npz']
        self.results.to_csv(self.gen_path('ensembleSelection')+'.csv', index=False)
        return (final_sc, final_sc_std, final_model, sumP)
