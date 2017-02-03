"""
class <FeatureProcessor>

usage: applies the following techniques to a pandas dataframe
1- replace missing values
2- feature normalization
3- one hot encoding
4- log transfrom
5- pairwise elimination
6- three-way and two-way features


how to use:


import pandas as pd
from FeatureProcessor import FeatureProcessor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.drop(['target'], axis=1, inplace=True)
test.drop(['target'], axis=1, inplace=True)

x = FeatureProcessor()
y = x.fit_transform(train, True, True, True, True)
z = x.transform(test)

"""

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import scipy.sparse as sp 
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    
    def __init__(self):
        self.init()


    def init(self):
        self.catList = []
        self.numList = []
        self.normDic = {}
        self.toBeRemoved = []
        self.numFeat = 0
        self.dictVectorizer = DictVectorizer(sparse = True)
        self.applyLog = False
        self.pwe = False
        self.tWay = False
        self.thrWay = False
        self.apply_svd = True
        self.fit = True
        self.size = 0
        self.numData = []
        self.catData = []
        self.two_way_list = []
        self.three_way_list = []


    def apply_trans(self):
        self.separate_data()

        self.log_transform()
        
        
        self.handle_missing()

        self.generate_svd()
        
        self.two_way()
        self.three_way()
        
        self.pairwise_elimination()

        self._standard_scaler()
        
        self.one_hot()
        
        if self.numData.shape[1] == 0 and self.catData.shape[1] != 0:
            self.data = self.catData
        elif self.numData.shape[1] != 0 and self.catData.shape[1] == 0:
            self.data = self.numData
        else:
            self.data = sp.csr_matrix(sp.hstack((sp.csr_matrix(self.numData), self.catData)))
        
    def separate_data(self):
        catList = []
        numList = []
        if self.fit == True:
            for f in self.data.columns:
                if self.data[f].dtype == 'object':
                    self.catList.append(f)
                else:
                    self.numList.append(f)

        
        self.catData = self.data[self.catList]
        self.catData = self.catData.fillna( 'NA' )

        self.numData = self.data[self.numList]
        self.sz = self.numData.shape[1]


    def fit_transform(self, data, applyLog, thrWay, tWay, pwe, apply_svd, target):
        
        """
        @description: fits the class to the new data then transform
        the data.

        @param:
        data:       pandas dataframe
        applyLog:    bool     ==> call log_transform()?
        thrWay:     bool     ==> call three_way()?
        tWay:         bool     ==> call two_way()?
        pwe:         bool    ==> call pairwise_elimination() ? 

        """
        self.data = data.copy()
        self.init()
        #print self.data.shape
        self.size = len(self.data.columns)

        self.applyLog = applyLog
        self.pwe = pwe
        self.tWay = tWay
        self.thrWay = thrWay
        self.apply_svd = apply_svd
        self.target = target

        self.apply_trans()
        
        #print 'FINISHED'
        #print self.data.shape
        return self.data



    def transform(self, data):
        
        """
        @description: transforms the data according to a previously fitted data.
        
        @param:
        data:       pandas dataframe
        """
        
        self.data = data.copy()
        self.fit = False

        self.apply_trans()
        
        return self.data



    def _min_max_norm(self):
        # Does min-max normalization and assumes that missing value = -999
        for f in self.numData:
            if self.fit == True:
                minVal = np.nanmin(self.numData[f].as_matrix())
                maxVal = np.nanmax(self.numData[f].as_matrix())
                self.normDic[f] = (minVal, maxVal)
            else:
                minVal = self.normDic[f][0]
                maxVal = self.normDic[f][1]
            
            # min max normalization.
            self.numData[f] = (self.numData[f] - minVal) / maxVal
            
    def _standard_scaler(self):
        if self.fit == True:
            self.scaler = StandardScaler()
            self.scaler.fit(self.numData)
        self.numData = self.scaler.transform(self.numData)
        

    def handle_missing(self):
        self._min_max_norm()
        self.numData.fillna(-999, inplace=True)
        self.numData = self.numData.as_matrix()

    def one_hot(self):
        
        self.catData = self.catData.T.to_dict().values()    

        if self.fit==True:
            self.catData = self.dictVectorizer.fit_transform(self.catData)
        else:
            self.catData = self.dictVectorizer.transform(self.catData)


    def log_transform(self):

        # check if a column has skewed data
        # if so then apply log transfrom to that column.
        
        if self.applyLog == True:
            if self.fit == True:
                # calculate skew metric for all columns.
                self.isSkewed = scipy.stats.skew(self.numData, axis=0, bias=True)
            
            self.numData = self.numData + np.full(self.numData.shape, 0.005)
            for col in range(0,self.numData.shape[1]):
                if self.isSkewed[col] > 2 or self.isSkewed[col] < -2:
                    self.numData.iloc[:,col] = np.log(self.numData.iloc[:,col])


    def two_way(self):

        # multiply all pairs and add result to matrix as new features.
        sz=self.sz
        if self.tWay == True:
            if self.fit == True:
                for i in range(0, sz-1):
                    for j in range (i+1, sz):
                        #print i,j,sz,'2-WAY ---- SEBO SHA3\'AAAAL'
                        newCol = np.multiply(self.numData[:,i], self.numData[:,j])
                        newCol[newCol==-0]=0

                        temp = np.zeros((newCol.shape[0],1))
                        for m in range(newCol.shape[0]):
                            temp[m] = newCol[m]
                        temp = np.matrix(temp)
                        f, _ = f_classif(temp, self.target)
                        if f[0] >= 1:
                            self.two_way_list.append((i,j))
                            self.numData = np.column_stack((self.numData, newCol))
            else:
                for i in range (len(self.two_way_list)):

                    newCol = np.multiply(self.numData[:,self.two_way_list[i][0]], 
                        self.numData[:,self.two_way_list[i][1]])

                    self.numData = np.column_stack((self.numData, newCol))
            #print 'FINISHED 2-Way'

        
    
    def three_way(self):        

        #print 'Three Way Start'
        # multiply all pairs and add result to matrix as new features.
        #sz = self.numData.shape[1]
        sz=self.sz
        if self.thrWay == True:
            if self.fit == True:
                for i in range(0, sz-2):
                    for j in range (i+1, sz-1):
                        tmp = np.multiply(self.numData[:,i], self.numData[:,j])
                        for k in range (j+1, sz):
                            #print i,j,k,sz, '3-WAY ---- SEBO SHA3\'AAAAL'
                            newCol = np.multiply(tmp, self.numData[:,k])
                            newCol[newCol==-0]=0

                        temp = np.zeros((newCol.shape[0],1))
                        for m in range(newCol.shape[0]):
                            temp[m] = newCol[m]
                        temp = np.matrix(temp)
                        f, _ = f_classif(temp, self.target)
                        if f[0] >= 1:
                            self.three_way_list.append((i,j,k))
                            self.numData = np.column_stack((self.numData, newCol))
            else:
                for i in range (len(self.three_way_list)):

                    newCol = np.multiply(self.numData[:,self.three_way_list[i][0]], 
                        self.numData[:,self.three_way_list[i][1]])

                    newCol = np.multiply(newCol, self.numData[:,self.three_way_list[i][2]])

                    self.numData = np.column_stack((self.numData, newCol))
            #print 'FINISHED 3-Way'

    def pairwise_elimination(self):
        
        # check covariance between every pair of variables
        # if covariance > 0.8 then remove one of these variables.
        sz = self.sz
        if self.pwe == True:
            if self.fit == True:
                for i in range(0,  sz):
                    for j in range(i+1, sz):
                        cov = np.corrcoef(self.numData[:,i], self.numData[:,j])[1,0]
                        if cov > 0.8:
                            if not (j in self.toBeRemoved):
                                self.toBeRemoved.append(j)
            self.numData = np.delete(self.numData, self.toBeRemoved, 1)
            
    def generate_svd(self):
        if self.apply_svd == True:
            if self.fit == True:
                self.svd = TruncatedSVD(n_components = 3)
                self.svd.fit(self.numData)
            self.numData = np.hstack((self.numData, self.svd.transform(self.numData)))