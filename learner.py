from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree
import xgboost as xgb
from multiprocessing import cpu_count
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

class ClassificationCV(object):
    def cv(self, X, y, kf):
        n_classes = len(np.unique(y))
        p = np.zeros((X.shape[0], n_classes))
        for train, test in kf:
            self.fit(X[train], y[train])
            p[test] += self.predict_proba(X[test])
            if test.shape[0] > train.shape[0]:
                p[test] /= len(kf) - 1
        return p

class RegressionCV(object):
    def cv(self, X, y, kf):
        p = np.zeros(X.shape[0])
        for train, test in kf:
            self.fit(X[train], y[train])
            p[test] += self.predict(X[test])
            if test.shape[0] > train.shape[0]:
                p[test] /= len(kf) - 1
        return p
    
# Neural Network

class NeuralNetworkClassifier(ClassificationCV):
    space = {'units': ('uniform', 2, 512, 'discrete'), 'layers': ('uniform', 1, 5, 'discrete'), 'dropout': ('quniform', 0.1, 0.9, 0.05, 'continuous'), 'batch_size' : ('uniform', 28, 128, 'discrete')}
    parallel_cv = False
    def __init__(self, units=None, layers=1, dropout=0.5, loss='categorical_crossentropy', init='normal', activation='relu', final_activation='sigmoid', nb_epoch=10, batch_size=32, verbose=0):
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.loss = loss
        self.init = init
        self.activation = activation
        self.final_activation = final_activation
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.units, input_dim=self.input_dim, init=self.init, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        for i in range(1, self.layers):
            model.add(Dense(self.units, init=self.init, activation=self.activation))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))
        model.add(Dense(self.num_class, init=self.init, activation=self.final_activation))
        model.compile(loss=self.loss, optimizer='adam')
        return model
        
    def fit(self, X, y):
        self.num_class = len(np.unique(y))
        # if self.num_class > 2:
        self.loss = 'categorical_crossentropy'
        Y = np_utils.to_categorical(y)
        # else:
            # self.loss = 'binary_crossentropy'
            # Y = y
        self.input_dim = X.shape[1]
        if self.units is None:
            self.units = X.shape[1]
        # self.model = self.build_model()
        self.model = Sequential()
        self.model.add(Dense(self.units, input_dim=self.input_dim, init=self.init, activation=self.activation))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout))
        for i in range(1, self.layers):
            self.model.add(Dense(self.units, init=self.init, activation=self.activation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.num_class, init=self.init, activation=self.final_activation))
        self.model.compile(loss=self.loss, optimizer='adam')
        return self.model.fit(X, Y, nb_epoch=self.nb_epoch, batch_size=self.batch_size, verbose=self.verbose)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    # def set_params(self, **params):
        # if not params:
            # # Simple optimisation to gain speed (inspect is slow)
            # return self
        # for key, value in params.iteritems():
            # if key not in self.params:
                # raise ValueError('Invalid parameter %s for estimator %s. ' % (key, self.__class__.__name__))
            # self.params[key] = value
        # return self
    
# K Nearest Neighbors

class KNeighborsClassifier(neighbors.KNeighborsClassifier, ClassificationCV):
    grid = {'n_neighbors':[3, 5, 11, 25, 49, 99]}
    space = {'n_neighbors': ('uniform', 1, 'samples', 'discrete')}
    parallel_cv = False
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs):
        super(KNeighborsClassifier, self).__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        
class KNeighborsRegressor(neighbors.KNeighborsRegressor, RegressionCV):
    grid = {'k':[3, 5, 11, 25, 49, 99]}
    space = {'n_neighbors': ('uniform', 1, 'samples', 'discrete')}
    parallel_cv = False
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs):
        super(KNeighborsRegressor, self).__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs, **kwargs)

# Logistic Regression

class LogisticRegression(linear_model.LogisticRegression, ClassificationCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        super(LogisticRegression, self).__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        
# Linear Regression

class LassoRegression(linear_model.Lasso, RegressionCV):
    grid = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
    space = {'alpha': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        super(LassoRegression, self).__init__(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)

class RidgeRegression(linear_model.Ridge, RegressionCV):
    grid = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
    space = {'alpha': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
        super(RidgeRegression, self).__init__(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver, random_state=random_state)
        
# Naive Bayes

class GaussianNB(naive_bayes.GaussianNB, ClassificationCV):
    def __init__(self, priors=None):
        super(GaussianNB, self).__init__(priors=priors)

# Multinomial Naive Bayes

class MultinomialNB(naive_bayes.MultinomialNB, ClassificationCV):
    grid = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
    space = {'alpha': ('quniform', 0.1, 2, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(MultinomialNB, self).__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
 
# Bernoulli Naive Bayes
       
class BernoulliNB(naive_bayes.BernoulliNB, ClassificationCV):
    grid = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}
    space = {'alpha': ('quniform', 0.1, 2, 0.1, 'continuous')}  
    parallel_cv = False
    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True, class_prior=None):
        super(BernoulliNB, self).__init__(binarize=binarize, fit_prior=fit_prior, class_prior=class_prior)

# Linear SVM

class Linear_SVC(svm.SVC, ClassificationCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, C=1.0, shrinking=True, probability=True, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None):
        super(Linear_SVC, self).__init__(kernel='linear', C=C, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)
        
class Linear_SVR(svm.SVR, RegressionCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, epsilon=0.1, C=1.0, shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1):
        super(Linear_SVR, self).__init__(kernel='linear', epsilon=epsilon, C=C, shrinking=shrinking, tol=tol, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        
# Polynomial SVM        
        
class Poly_SVC(svm.SVC, ClassificationCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],'gamma':[0.01, 0.001, 0.0001],'degree':[2,3]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous'), 'gamma': ('qloguniform', -3, 3, 0.1, 'continuous'), 'degree': ('uniform', 1, 5, 'discrete')}
    parallel_cv = False
    def __init__(self, C=1.0, degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None):
        super(Poly_SVC, self).__init__(kernel='poly', C=C, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)
        
class Poly_SVR(svm.SVR, RegressionCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],'gamma':[0.01, 0.001, 0.0001],'degree':[2,3]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous'), 'gamma': ('qloguniform', -3, 3, 0.1, 'continuous'), 'degree': ('uniform', 1, 5, 'discrete')}
    parallel_cv = False
    def __init__(self, epsilon=0.1, C=1.0, degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1):
        super(Poly_SVR, self).__init__(kernel='poly', epsilon=epsilon, C=C, shrinking=shrinking, tol=tol, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        
# Radial Basis Function SVM       
        
class RBF_SVC(svm.SVC, ClassificationCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],'gamma':[0.01, 0.001, 0.0001]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous'), 'gamma': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, C=1.0, degree=3, gamma='auto', shrinking=True, probability=True, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None):
        super(RBF_SVC, self).__init__(kernel='rbf', C=C, degree=degree, gamma=gamma, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)
        
class RBF_SVR(svm.SVR, RegressionCV):
    grid = {'C':[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],'gamma':[0.01, 0.001, 0.0001]}
    space = {'C': ('qloguniform', -3, 3, 0.1, 'continuous'), 'gamma': ('qloguniform', -3, 3, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, epsilon=0.1, C=1.0, degree=3, gamma='auto', shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1, ):
        super(RBF_SVR, self).__init__(kernel='rbf',  epsilon=epsilon, C=C, shrinking=shrinking, tol=tol, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

# Decision Tree

class DecisionTreeClassifier(tree.DecisionTreeClassifier, ClassificationCV):
    grid = {'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto', 'sqrt', 0.2]}
    space = {'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}
    parallel_cv = False
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False):
        super(DecisionTreeClassifier, self).__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, class_weight=class_weight, presort=presort)
        
class DecisionTreeRegressor(tree.DecisionTreeRegressor, RegressionCV):
    grid = {'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto', 'sqrt', 0.2]}
    space = {'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}
    parallel_cv = False
    def __init__(self, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False):
        super(DecisionTreeRegressor, self).__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, presort=presort)
        
# AdaBoost

class AdaBoostClassifier(ensemble.AdaBoostClassifier, ClassificationCV):
    grid = {'n_estimators':[100, 200, 400], 'learning_rate':[0.01, 0.1, 1], 'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto', 'sqrt', 0.2]}
    space = {'n_estimators': ('uniform', 1, 1000, 'discrete'), 'learning_rate': ('quniform', 0.01, 1, 0.05, 'continuous'), 'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}
    parallel_cv = False
    def __init__(self, base_estimator=tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False), n_estimators=50, learning_rate=1., algorithm='SAMME.R', random_state=None):
        super(AdaBoostClassifier, self).__init__(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)
            
class AdaBoostRegressor(ensemble.AdaBoostRegressor, RegressionCV):
    grid = {'n_estimators':[100, 200, 400], 'learning_rate':[0.01, 0.1, 1], 'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto', 'sqrt', 0.2]}
    space = {'n_estimators': ('uniform', 1, 1000, 'discrete'), 'learning_rate': ('quniform', 0.01, 1, 0.05, 'continuous'), 'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}
    parallel_cv = False
    def __init__(self, base_estimator=tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False), n_estimators=50, learning_rate=1., loss='linear', random_state=None):
        super(AdaBoostRegressor, self).__init__(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=random_state)
    
# Random Forest

class RandomForestClassifier(ensemble.RandomForestClassifier, ClassificationCV):
    grid = {'n_estimators':[100, 200, 400], 'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto','sqrt', 0.2]}
    space = {'n_estimators': ('uniform', 1, 1000, 'discrete'), 'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}
    parallel_cv = False
    def __init__(self, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None):
        super(RandomForestClassifier, self).__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)
            
class RandomForestRegressor(ensemble.RandomForestRegressor, RegressionCV):
    grid = {'n_estimators':[100, 200, 400], 'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto','sqrt', 0.2]}
    space = {'n_estimators': ('uniform', 1, 1000, 'discrete'), 'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}
    parallel_cv = False
    def __init__(self, n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False):
        super(RandomForestRegressor, self).__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
        
# ExtraTrees

class ExtraTreesClassifier(ensemble.ExtraTreesClassifier, ClassificationCV):
    grid = {'n_estimators':[100, 200, 400], 'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto','sqrt', 0.2]}
    space = {'n_estimators': ('uniform', 1, 1000, 'discrete'), 'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')} 
    parallel_cv = False    
    def __init__(self, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None):
        super(ExtraTreesClassifier, self).__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)
            
class ExtraTreesRegressor(ensemble.ExtraTreesRegressor, RegressionCV):
    grid = {'n_estimators':[100, 200, 400], 'max_depth':[3, 7, 11], 'min_samples_leaf':[1, 50, 100], 'min_samples_split':[1, 2, 7], 'max_features':['auto','sqrt', 0.2]}
    space = {'n_estimators': ('uniform', 1, 1000, 'discrete'), 'max_features': ('quniform', 0.05, 1, 0.05, 'continuous'), 'min_samples_leaf': ('quniform', 1, 500, 5, 'continuous'), 'min_samples_split': ('quniform', 1, 500, 5, 'continuous'), 'max_depth': ('uniform', 5, 15, 'discrete')}   
    parallel_cv = False
    def __init__(self, n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False):
        super(ExtraTreesRegressor, self).__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
            
# XGBoost Linear

class XGBLinearClassifier(ClassificationCV):
    space = {'num_round' : ('uniform', 100, 1000, 'discrete'), 'eta' : ('quniform', 0.01, 1, 0.05, 'continuous'), 'reg_lambda' : ('quniform', 0, 5, 0.5, 'continuous'), 'reg_alpha' : ('quniform', 0, 0.5, 0.05, 'continuous'), 'lambda_bias' : ('quniform', 0, 3, 0.5, 'continuous')} 
    parallel_cv = False
    def __init__(self, silent=1, n_jobs=1, eta=0.1, reg_lambda=1, reg_alpha=0, lambda_bias=0, objective="multi:softprob", base_score=0.5, error="error", seed=0, num_round=100):
        self.params = {"booster":"gblinear", "silent":silent, "nthread":n_jobs, "eta":eta, "lambda":reg_lambda, "alpha":reg_alpha, "lambda_bias":lambda_bias, "objective":objective, "base_score":base_score, "error":error, "seed":seed}
        self.num_round = num_round
            
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        if self.params['objective'] == 'binary:logistic' or self.params['objective'] == "binary:logitraw":
            self.params['num_class'] = 1
        else:
            self.params['num_class'] = len(np.unique(y))
        self.model = xgb.train(self.params, dtrain, self.num_round)
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        p = self.model.predict(dtest)
        return np.argmax(p, axis=1)
        
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        p = self.model.predict(dtest)
        if self.params['objective'] == 'binary:logistic':
            p = np.column_stack([1-p, p])
        return p
        
    def set_params(self, **params):
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        for key, value in params.iteritems():
            if key not in self.params:
                raise ValueError('Invalid parameter %s for estimator %s. ' % (key, self.__class__.__name__))
            self.params[key] = value
        return self
        
class XGBLinearRegressor(RegressionCV):
    space = {'num_round' : ('uniform', 100, 1000, 'discrete'), 'eta' : ('quniform', 0.01, 1, 0.05, 'continuous'), 'reg_lambda' : ('quniform', 0, 5, 0.05, 'continuous'), 'reg_alpha' : ('quniform', 0, 0.5, 0.05, 'continuous'), 'lambda_bias' : ('quniform', 0, 3, 0.5, 'continuous')} 
    parallel_cv = False
    def __init__(self, silent=1, n_jobs=1, eta=0.1, reg_lambda=1, reg_alpha=0, lambda_bias=0, objective="reg:linear", base_score=0.5, error="rmse", seed=0, num_round=100):
        self.params = {"booster":"gblinear", "silent":silent, "nthread":n_jobs, "eta":eta, "lambda":reg_lambda, "alpha":reg_alpha, "lambda_bias":lambda_bias, "objective":objective, "base_score":base_score, "error":error, "seed":seed}
        self.num_round = num_round
            
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(self.params, dtrain, self.num_round)
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        p = self.model.predict(dtest)
        return np.argmax(p, axis=1)
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def set_params(self, **params):
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        for key, value in params.iteritems():
            if key not in self.params:
                raise ValueError('Invalid parameter %s for estimator %s. ' % (key, self.__class__.__name__))
            self.params[key] = value
        return self
   
# XGBoost Tree   

class XGBTreeClassifier(ClassificationCV):
    space = {'num_round' : ('uniform', 100, 1000, 'discrete'), 'eta' : ('quniform', 0.01, 1, 0.05, 'continuous'), 'max_depth' : ('uniform', 1, 15, 'discrete'), 'min_child_weight' : ('quniform', 1, 10, 1, 'continuous'), 'subsample' : ('quniform', 0.5, 1, 1, 'continuous'), 'gamma' : ('quniform', 0.05, 2, 0.1, 'continuous'), 'colsample_bytree' : ('quniform', 0.1, 1, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, silent=1, n_jobs=1, eta=0.1, gamma=0, max_depth=3, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_lambda=1, reg_alpha=0, tree_method="auto", sketch_eps=0.3, scale_pos_weight=1, objective="multi:softprob", base_score=0.5, error="error", seed=0, num_round=100):
        self.params = {"silent":silent, "nthread":n_jobs, "eta":eta, "gamma":gamma, "max_depth":max_depth, "min_child_weight":min_child_weight, "max_delta_step":max_delta_step, "subsample":subsample, "colsample_bytree":colsample_bytree, "colsample_bylevel":colsample_bylevel, "lambda":reg_lambda, "alpha":reg_alpha, "tree_method":tree_method, "sketch_eps":sketch_eps, "scale_pos_weight":scale_pos_weight, "objective":objective, "base_score":base_score, "error":error, "seed":seed}
        self.num_round = num_round
            
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        if self.params['objective'] == 'binary:logistic' or self.params['objective'] == "binary:logitraw":
            self.params['num_class'] = 1
        else:
            self.params['num_class'] = len(np.unique(y))
        self.model = xgb.train(self.params, dtrain, self.num_round)
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        p = self.model.predict(dtest)
        return np.argmax(p, axis=1)
        
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        p = self.model.predict(dtest)
        if self.params['objective'] == 'binary:logistic' or self.params['objective'] == "binary:logitraw":
            p = np.column_stack([1-p, p])
        return p
        
    def set_params(self, **params):
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        for key, value in params.iteritems():
            if key not in self.params:
                raise ValueError('Invalid parameter %s for estimator %s. ' % (key, self.__class__.__name__))
            self.params[key] = value
        return self
        
class XGBTreeRegressor(RegressionCV):
    space = {'num_round' : ('uniform', 100, 1000, 'discrete'), 'eta' : ('quniform', 0.01, 1, 0.05, 'continuous'), 'max_depth' : ('uniform', 1, 15, 'discrete'), 'min_child_weight' : ('quniform', 1, 10, 1, 'continuous'), 'subsample' : ('quniform', 0.5, 1, 1, 'continuous'), 'gamma' : ('quniform', 0.05, 2, 0.1, 'continuous'), 'colsample_bytree' : ('quniform', 0.1, 1, 0.1, 'continuous')}
    parallel_cv = False
    def __init__(self, silent=1, n_jobs=1, eta=0.1, gamma=0, max_depth=3, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_lambda=1, reg_alpha=0, tree_method="auto", sketch_eps=0.3, scale_pos_weight=1, objective="reg:linear", base_score=0.5, error="rmse", seed=0, num_round=100):
        self.params = {"silent":silent, "nthread":n_jobs, "eta":eta, "gamma":gamma, "max_depth":max_depth, "min_child_weight":min_child_weight, "max_delta_step":max_delta_step, "subsample":subsample, "colsample_bytree":colsample_bytree, "colsample_bylevel":colsample_bylevel, "lambda":reg_lambda, "alpha":reg_alpha, "tree_method":tree_method, "sketch_eps":sketch_eps, "scale_pos_weight":scale_pos_weight, "objective":objective, "base_score":base_score, "error":error, "seed":seed}
        self.num_round = num_round
            
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(self.params, dtrain, self.num_round)
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def set_params(self, **params):
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        for key, value in params.iteritems():
            if key not in self.params:
                raise ValueError('Invalid parameter %s for estimator %s. ' % (key, self.__class__.__name__))
            self.params[key] = value
        return self
