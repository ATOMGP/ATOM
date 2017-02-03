import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from pylatex import Document, Section, Subsection,Subsubsection, Figure, Tabular, Math, \
                    TikZ, Axis, Plot, Figure, Package, Matrix, Command, \
                    NoEscape, MultiColumn, Tabu, MultiRow, Itemize
from pylatex.utils import italic
import pandas as pd
import os
import re
import random
from sklearn.preprocessing import label_binarize
from visualizer import visualizer

class ReportGenerator():
    '''
    - a class to generate the report (pdf only).
    - uses the methods from visualization class to draw graphs.
    - takes as input: path, options user selected, final model
    '''

    def __init__(self, X_train, y_train, target_names, path, experiments, maximize,
                    num_of_iter, regression, elapsed_time, final_model_time):
        # assume model is an object contains function to predict #
        self.doc = Document(path)
        self.experiments = experiments
        # sort experiements on score
        self.experiments = self.experiments.sort_values('Score Mean', ascending = not maximize)
        print '----------- Experiments -------------\n'
        print [p for p in self.experiments['Learner']]
        self.num_of_iter = num_of_iter
        self.num_of_features= X_train.shape[1]
        self.regression = regression
        self.elapsed_time = elapsed_time
        self.final_model_time = final_model_time
        self.X_train = X_train
        self.y_train = y_train
        self.target_names = target_names
        self.path = path
        self.probabilities = np.load(self.experiments['Path'].iloc[0])['PREDICTIONS']
        if not regression:
            self.predictions = np.argmax(self.probabilities, axis = 1)
        
    # function checks if the given model is ensemble #
    def is_ensemble(self, s):
        return (s[:2] == "es")

    # function prints single model's parameters #
    def print_param(self, row):
        self.doc.append(self.format_dict(row[3]) + '\n')

    # functions extracts the actual model name to be shown #
    # in the doc from the encodded one #
    def get_model_name(self, s):
        ns = ''
        if self.is_ensemble(s):
            ns = re.sub("es", "Ensemble Selection", s)
            ns = re.sub('[_]', ' ', ns)
        else:
            for c in s:
                if c == c.upper():
                    ns += ' '
                ns += c
        return ns
    
    #edit function print ensemble
    def print_ensemble(self, row):
        d = row[3] # dictionary of dictionaries(of ensembles)
        cnt = 1
        with self.doc.create(Subsubsection('Ensemble Selection', numbering=False)):
            self.doc.append(NoEscape(r'\leftskip=40pt')) # indentation
            self.doc.append('Score: ' + str(row[1])  + '\n\n')
            for sub_d in d:
                self.doc.append('Bag: ' + str(cnt) + '\n\n')
                cnt += 1
                # for every ensemble print it
                table = Tabular('|c|c|c|l|')
                table.add_hline()
                # add header of table #
                table.add_row(('Learner', 'Score', 'Parameters','weight'))
                table.add_hline()
                for k in sub_d:
                    cur_model = self.experiments.loc[k]
                    data = [cur_model[0], round(cur_model[1],4), self.format_dict(cur_model[3]), sub_d[k]]
                    table.add_row(data)
                    table.add_hline()
                self.doc.append(table)
                self.doc.append('\n\n\n\n')

            
    # function prints the layer zero models of the given ensemble model
    def print_ensemble_models(self, row):
        # - create a sub_sub_section to easily indent #
        # - subsubsection title is the ensmble name #
        # - table of the ensemble models (layer 0) #
        ensemble_method = row[0] # name of the ensmble method
        with self.doc.create(Subsubsection(self.get_model_name(ensemble_method), numbering=False)):
            # create  table for the ensmeble models #
            self.doc.append(NoEscape(r'\leftskip=40pt')) # indentation
            self.doc.append('Score: ' + str(row[1])  + '\n\n')
            table = Tabular('|c|c|c|l|')
            table.add_hline()
            # add header of table #
            table.add_row(('Learner', 'Score', 'Parameters','weight'))
            table.add_hline()
            # foreach model in the ensemble add row in the table #
            for k in row[3]:
                cur_model = self.experiments.loc[k]
                data = [cur_model[0], cur_model[1], self.format_dict(cur_model[3]), row[3][k]]
                table.add_row(data)
                table.add_hline()
            self.doc.append(table)

    # function converts dictionary of parameters into a string #
    def format_dict(self, d):
        if type(d) == type(''):
            d = eval(d)
        s = ""
        f =  False
        for k in d:
            if f:
                s +=  ", "
            f = True
            s += (str(k) + ":" + str(d[k]))
        return s

    def gen_summary(self, score):
        '''
        - function generates the first part of the doc the summary of the final model
        - inputs are booleans to decide whether to show then in the report or not.
        '''
        with self.doc.create(Section('Summary', numbering=False)):
            # -------- Final Model Description --------#
            '''
            final model:- single: learner name, parameters
                        - ensemble: type, models(parameters, scores)
            '''
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Final Model Description', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                # check if ensemble or single model from its name #
                #edit
                #if self.is_ensemble(self.experiments.iloc[0][0]):
                #    self.print_ensemble_models(self.experiments.iloc[0])
                if self.experiments.iloc[0][0] == "ensembleSelection":
                    self.print_ensemble(self.experiments.iloc[0])
                else:
                    model_name = self.get_model_name(self.experiments.iloc[0][0])
                    self.doc.append(model_name + ": ")
                    self.print_param(self.experiments.iloc[0])

            # ----------- Number OF iterations -----------#
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Number of iterations', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                self.doc.append(str(self.num_of_iter))

            # ----------- Number Of Features -------------#
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Number of features', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                self.doc.append(str(self.num_of_features))

            # ---------- Classification / Regression ------#
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Task type', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                if self.regression:
                    self.doc.append('Regression')
                else:
                    self.doc.append('Classification')
            '''
            # ----------- Elapsed Time ------------------#
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Elapsed Time', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                self.doc.append(self.elapsed_time)

            # --------------- final model time -------- #
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Final Model Time', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                self.doc.append(self.final_model_time)
            '''

    # function generates a table of the best models #
    def draw_top_models(self, n):
        '''
        - functions draw a table of the top models, with theri details: name, score, parameters.
        - takes the data frame of the models as input.
        - print the best ensemble models, then the best single models in a table
        '''
        self.doc.append(NoEscape(r'\leftskip=0pt'))
        with self.doc.create(Section('Top' + ' ' + str(n) + ' ' + 'Models',numbering = False)):
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            single_models_table = Tabular("|c|c|c|")
            single_models_table.add_hline()
            single_models_table.add_row(["learner", "Score", "Parameters"])
            single_models_table.add_hline()
            # if ensemble print it, else append to the table
            k = 0
            single = 0
            ens = 0
            for model in self.experiments.values:
                if k >= n:
                    break
                print 'Model---\n', model[0]
                #edit
                if model[0] != "ensembleSelection":
                    #self.doc.append(NoEscape(r'\leftskip=20pt'))
                    #self.print_ensemble(model)
                #else:
                    data = [model[0],model[1], self.format_dict(model[3])]
                    single_models_table.add_row(data)
                    single_models_table.add_hline()
                    single += 1
                    k += 1
            if single > 0:
                self.doc.append(NoEscape(r'\leftskip=20pt'))
                with self.doc.create(Subsubsection('Single Models',numbering = False)):
                    self.doc.append(NoEscape(r'\leftskip=40pt'))
                    self.doc.append(single_models_table)

    # function generates the graphs according to user preferences #
    # uses visualization class #
    def gen_graphs(self, visu):
        '''
        function calls the graphs methods to draw from the visualization
        module according to use preferences
        takes as input a visualizer object
        '''
        # start Graphs Section in the report #
        self.doc.append(NoEscape(r'\leftskip=0pt'))
        with self.doc.create(Section('Graphs', numbering=False)):
            # uncomment after generating predictions
            
            if not self.regression:
                # Confusion Matrix #
                self.doc.append(NoEscape(r'\leftskip=20pt'))
                with self.doc.create(Subsection('Confusion Matrix', numbering=False)):
                    self.doc.append(NoEscape(r'\leftskip=40pt'))
                    with self.doc.create(Figure(position='htbp')) as conf_mat_plot: # Create new Figure in tex
                        # get image path from the visualizer
                        file_name = visu.gen_conf_mat(target_names = self.target_names, predictions = self.predictions,
                                                      y = self.y_train,dpi = 300)
                        conf_mat_plot.add_image(file_name, width=NoEscape(r'1\textwidth'))
                
                self.doc.append(NoEscape(r'\pagebreak')) #start new page
                # ROC Curve #
                self.doc.append(NoEscape(r'\leftskip=20pt'))
                with self.doc.create(Subsection('ROC Curve', numbering=False)):
                    self.doc.append(NoEscape(r'\leftskip=40pt'))
                    with self.doc.create(Figure(position='htbp')) as ROC_plot: # Create new Figure in tex
                        # y_tr = self.y_train
                        y_ts = self.y_train
                        file_name = visu.gen_roc_curve(probabilities = self.probabilities, y = y_ts, target_names = 
                        self.target_names)
                        ROC_plot.add_image(file_name, width=NoEscape(r'1.2\textwidth'))
                self.doc.append(NoEscape(r'\pagebreak')) #start new page
                        
            # Feature importance #
            self.doc.append(NoEscape(r'\leftskip=20pt'))
            with self.doc.create(Subsection('Feature Importance', numbering=False)):
                self.doc.append(NoEscape(r'\leftskip=40pt'))
                with self.doc.create(Figure(position='htbp')) as imp_plot:
                    file_name = visu.gen_feature_imp(regression = self.regression, X = self.X_train, y = self.y_train)
                    imp_plot.add_image(file_name, width=NoEscape(r'1.2\textwidth'))
             
            self.doc.append(NoEscape(r'\pagebreak')) #start new page
            if self.experiments.iloc[0][0] == 'ensembleSelection':
                # Word Cloud #
                self.doc.append(NoEscape(r'\leftskip=20pt'))
                with self.doc.create(Subsection('Models Cloud', numbering=False)):
                    self.doc.append(NoEscape(r'\leftskip=40pt'))
                    with self.doc.create(Figure(position='htbp')) as cloud_plot:
                        file_name = visu.gen_word_cloud(self.experiments)
                        cloud_plot.add_image(file_name, width=NoEscape(r'1.2\textwidth'))
                 
    # main function to be called to generate the report #
    def generate(self):
        '''
        main function that generate the report and call the
        functions of the report sections
        '''
        self.doc.packages.append(Package('geometry', options=['tmargin=1cm',
                                                     'lmargin=0.5cm']))
        # cover page #
        self.doc.preamble.append(Command('title', 'Experiemts Report',))
        self.doc.preamble.append(Command('author', 'ATOM'))
        self.doc.append(NoEscape(r'\maketitle'))
        self.doc.append(NoEscape(r'\pagebreak'))
        # summary #
        self.gen_summary(self.experiments.iloc[0][1])
        self.doc.append(NoEscape(r'\pagebreak')) #start new page
        # Top N Models #
        self.draw_top_models(4)
        self.doc.append(NoEscape(r'\pagebreak')) #start new page
        # Graphs #
        visu = visualizer(save_dir = self.path) 
        self.gen_graphs(visu)
        # generate pdf file #
        self.doc.generate_pdf('example')
        print 'Finished Report Generation'

    

# --------------------------------- TESTING ------------------------------- #
def test():
    exp = gen_DF()
    #print exp
    #create newvisulaizer object
    iris = datasets.load_iris()
    X = iris.data
    
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = svm.SVC(kernel='linear', probability=True, random_state=True)

    #report = ReportGenerator(probabilities = classifier.fit(X_train, y_train).predict_proba(X_test),
    #                        X_train = X_train, y_train = y_train,
    #                        target_names = iris.target_names,
    #                        path = '/home/wesam/ATOM/atom/src/',
    #                        experiments = exp, num_of_iter= 0,
    #                        num_of_features = 0, regression = 0, elapsed_time= 0,
    #                        final_model_time = 0)
    #report.generate()

def gen_DF():
    results = pd.DataFrame(columns = ["learner", "Error", "Parameters"])
    name = 'ExtraTreesClassifier'
    score = 1.0
    param = {'c':1, 'gamma':2}
    results.loc[0] = ['es_with_replacement', 1.0, {3: 50, 4: 15, 5:1}]
    results.loc[1] = ['es_with_bagging', 1.0, {3: 2, 4: 3}]
    results.loc[2] = ['es_with_bagging', 1.0, {3: 2, 4: 3}]
    results.loc[3] = ['SVM', 3.0, {'c':1, 'gamma':15}]
    results.loc[4] = ['LogisticRegression', 3.0, {'C':1}]
    results.loc[5] = ['ExtraRandomTrees', 3.0, {'c':1, 'gamma':15}]
    return results

test()
