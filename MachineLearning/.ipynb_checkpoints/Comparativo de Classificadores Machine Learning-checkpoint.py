#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Comparação de CLassificadores
=====================

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs,make_gaussian_quantiles
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB

#pip install xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance

#pip install opfython
from opfython.models.knn_supervised import KNNSupervisedOPF
from opfython.models.supervised import SupervisedOPF
import opfython.math.general as g

#pip install sklearn-extensions
#conda install -c anaconda tensorflow-gpu

h = .02  # step size in the mesh

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

X, y = twospirals(1000)

names = ["Nearest Neighbors", "RBF SVM", "Decision Tree","Naive Bayes" 
         ,"OPF" 
         , "VotingClassifier","Random Forest", "AdaBoost", "XGBoost" ]


#opf = SupervisedOPF(distance = "log_squared_euclidean")
opf = SupervisedOPF()
#opf = KNNSupervisedOPF(max_k = 1)

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),    
    DecisionTreeClassifier(max_depth=10),
    GaussianNB(),
    opf,
    VotingClassifier(estimators=[('knn', KNeighborsClassifier(3)), ('svm',  SVC(gamma=2, C=1, probability=True)), ('gnb', GaussianNB()), ('dt', DecisionTreeClassifier(max_depth=10))], weights=[1.2,1.3,1.1,0.9], voting="soft", n_jobs = 4 ),
    RandomForestClassifier(max_depth=10, n_estimators=100),       
    AdaBoostClassifier(),
    XGBClassifier(n_estimators=20,max_depth=10)]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.4, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable,
            make_circles( noise=0.05, random_state=3),
            make_gaussian_quantiles(mean=(4, 4), cov=1, n_features=2, n_classes=2, random_state=1),
            make_classification(n_samples = 500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,class_sep=2,flip_y=0.2,weights=[0.5,0.5], random_state=17)
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        
        if name == "OPF":        
            X_train_opf, X_val_opf, y_train_opf, y_val_opf =  train_test_split(X_train, y_train, test_size=.3, random_state=42)
     
            
            y_train_opf = y_train_opf + 1
            y_val_opf  = y_val_opf + 1
            y_test_opf = y_test + 1
            
            clf.learn(X_train_opf, y_train_opf, X_val_opf, y_val_opf, n_iterations=20)
          
        else:
            clf.fit(X_train, y_train)
        
        if name == "OPF":    
            preds = clf.predict ( X_test )
            
            acc = g.opf_accuracy( y_test_opf , preds )
            score = acc
            print(score)
            print(accuracy_score(y_test_opf , preds))
        else:
            score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
       

        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(Xfull)            
        elif name == "OPF":
            predsopf = clf.predict (Xfull )
            predsopf = np.asarray(predsopf)
            predsopf = predsopf -1
            #predAux = np.ones(len(predsxx), dtype = int )
                        
            Z = np.asarray(predsopf)
        else:           
            #aux = classifiers[0].predict_proba(Xfull)[:, 1]
            Z = clf.predict_proba(Xfull)[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
