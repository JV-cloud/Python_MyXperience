# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:38:16 2019
@author: felip
"""
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

dados = iris.data

print(dados)

print(iris.feature_names)

print(iris.target_names)
#0 - setosa
#1 - versicolor
#2 - virginica

print(iris.target)

#Dataframe que armazena os atributos das especies de flores, normalmente chamamos esse tipo de dado de variável x
df_flores =pd.DataFrame(iris.data)
df_flores.columns=['sepal_Length','sepal_width','petal_Length','petal_width']


#Dataframe que armazena os nomes das especies de flores, normalmente chamamos esse tipo de dado de variável y
df_especies=pd.DataFrame(iris.target)
df_especies.columns=['especie']


X = StandardScaler().fit_transform(df_flores)

db = DBSCAN(eps=0.5, min_samples=3).fit(X)
labels = db.labels_
print(set(labels))


db = DBSCAN(eps=0.3, min_samples=3).fit(X)
labels = db.labels_
print(set(labels))


db = DBSCAN(eps=0.5, min_samples=6).fit(X)
labels = db.labels_
print(set(labels))


db = DBSCAN(eps=0.5, min_samples=5).fit(X)
labels = db.labels_
print(set(labels))

#
df_especies['cluster'] = db.labels_


y = df_especies
#Converte a coluna cluster e a coluna especie do dataframe  df_especies (copiado para y) para os vetores y_dbscan e y_real
y_dbscan = y.cluster.values
y_real = y.especie.values

#Cria um vetor de cores a serem utilzados na plotagem
colormap=np.array(['Red','green','blue'])
plt.figure(2)
#Adicona varios pontos na plotagaem utilizando as informações de sepal_Length e sepal_width como coordenadas dos pontos a serem plotados
plt.scatter(df_flores.sepal_Length, df_flores.sepal_width, c=colormap[df_especies.especie], s=100)
plt.title('Separação Real')
plt.show()

plt.figure(3)
#Faz a plotagem dos registros do primeiro cluster 0 (y_kmeans == 0) e utiliza as colunas de indice 0 e 1 (sepal_Length e sepal_width) para a plotagem dos pontos nas intruções x[y_kmeans == 0, 0] e x[y_kmeans == 0, 1]
plt.scatter(X[y_dbscan == 0, 0], X[y_dbscan == 0, 1], s = 20, c = 'blue', label = 'versicolour')
plt.scatter(X[y_dbscan == 1, 0], X[y_dbscan == 1, 1], s = 20, c = 'red', label = 'setosa')
plt.scatter(X[y_dbscan == -1, 0], X[y_dbscan == -1, 1], s = 20, c = 'black', label = 'Ruído')


plt.legend()
plt.title('Clusterização')
plt.show()
