from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from os import path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import f_classif
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

class visualizer():
    '''
    class that generate graphs to be used in the rport and visualizationin GUI.
    '''
    def __init__(self, save_dir):
        self.save_dir =  save_dir # the directory to save the images in
        
    # helper function plots the confusion matrix, #
    # to be called by gen_conf_mat #
    def plot_conf_mat(self, conf_mat, target_names, cmap=plt.cm.Blues):
        plt.figure()
        plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
        colorbar = plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        save_path = self.save_dir
        plt.savefig(save_path)
        colorbar.remove()
        return save_path+'.png'

    # Function to be called to get the confusion matrix #
    # returns path to image (confusion matrix) #
    def gen_conf_mat(self, target_names, predictions, y, *args, **kwargs):
        # get the predictions
        y_pred = predictions
        # compute confusion matrix
        conf_mat = confusion_matrix(y, y_pred)
        # Normalize
        conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        #plt.figure()
        return self.plot_conf_mat(conf_mat = conf_mat_normalized, target_names = target_names)

    # Function to be called to get the ROC curve #
    # returns path to image (ROC curve) #
    def gen_roc_curve(self, probabilities, y, target_names):
        y = label_binarize(y, [i for i in range(len(target_names))])
        n_classes = y.shape[1]
        if n_classes == 1:
            y_score = probabilities[:,1].reshape(-1,1)
        else:
            y_score = probabilities
        print 'y.shape', y.shape, y_score.shape
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        
        # Plot ROC curves for the multiclass problem

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 linewidth=2)

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        save_path = self.save_dir + 'b'
        plt.savefig(save_path)
        plt.show()
        return save_path+'.png'

                    
        
    # Function to be called to get the decision boundry of a classifier #
    # returns path to image #
    # model is not fitted #
    def gen_decision_boundary(self, model):
        n,m = self.X_train.shape
        # check if X has more than 2 dimensions, then embed #
        if m > 2:
            m = TSNE(n_components=2, random_state=0)
            X_embedded = m.fit_transform(self.X_train)
        else:
            X_embedded = self.X_train
        h = 0.2  # step size in the mesh
        classifier = model.fit(X_embedded, self.y_train)
        
        # create a mesh to plot in
        x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
        y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # predict
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        # Plot also the training points
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.y_train, cmap=plt.cm.Paired)
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        # need to check if not numbers #
        x_ticks = np.arange(x_min, x_max, (x_max - x_min)/10)
        y_ticks = np.arange(y_min, y_max, (y_max - y_min)/10)
        plt.xticks(x_ticks, [round(i) for i in x_ticks])
        plt.yticks(y_ticks, [round(i) for i in y_ticks])
        save_path = self.save_dir + 'a'
        plt.savefig(save_path)
        return save_path+'.png'
        
    # function computes #
    def gen_feature_imp(self, X, y, regression=0):
        # task_type: 0 classification #
        #            1 regression #
        if not regression:
            forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        else:
            forest = ExtraTreesRegressor(n_estimators=250, random_state=0)
        # X, y = self.X_train, self.y_train
        forest.fit(X, y)
        importances = forest.feature_importances_

        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        if len(indices) > 10:
            indices = [indices[i] for i in range(10)]
        # Plot the feature importances of the forest
        plt.figure()
        #edit
        plt.bar(range(len(indices)), importances[indices],
               color="r", align="center")
        plt.xticks(range(len(indices)), indices)
        plt.xlim([-1, len(indices)])
        plt.show()
        plt.ylabel('Importance')
        plt.xlabel('Features')
        
        save_path = self.save_dir + 'c'
        plt.savefig(save_path)
        return save_path+'.png'
    
    # Generate word cloud of ensemble models #
    # takes input sorted data frame of experiements #
    def gen_word_cloud(self, experiments):
        # write models' names in file #
        
        print experiments.iloc[0][3]
        list_models = experiments.iloc[0][3]
        
        with open('cloud.txt', 'w+') as file:
            for models in list_models:
                for k in models: # loop on the models of ensemble
                    if type(k) == type(''):
                        k = eval(k)
                    freq = models[k]
                    print k, models[k]
                    for i in range(freq): # write the model name
                        file.write(experiments.loc[k][0])
                        file.write('\n')
                        print experiments.loc[k][0]

        d = path.dirname(__file__)
        text = open(path.join(d, 'cloud.txt')).read() 
        wordcloud = WordCloud().generate(text)
        
        # Display the generated image:
        # the matplotlib way:
        import matplotlib.pyplot as plt
        plt.imshow(wordcloud)
        plt.axis("off")

        # take relative word frequencies into account, lower max_font_size
        wordcloud = WordCloud(max_font_size=50, relative_scaling=.5, background_color='white').generate(text)
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        save_path = self.save_dir + 'd'
        plt.savefig(save_path)
        return save_path+'.png'
    
    def gen_pair_wise_features():
        pass
