# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:38:16 2019
@author: felip
"""
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

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

#Escolha do valor aleatório 7
kmeans7 = KMeans(n_clusters=7)

#Executa o k-means com com k igual a 7 apenas para exemplo
kmeans7.fit(df_flores)

#Imprime os centros dos clusters
print(kmeans7.cluster_centers_)

print(kmeans7.labels_)


#Método Elbow para encontrar o melhor valor de k
inertias =[]
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i).fit(df_flores)    
    #somatório dos erros quadráticos das instâncias de cada cluster
    inertias.append(kmeans.inertia_)

    
plt.figure(1)
plt.plot(range(1, 15), inertias)
plt.title('O Metodo Elbow')
plt.xlabel('No de clusters')
plt.ylabel('WSS - within cluster sum of squares')
plt.show()

#O melhor k é igual a 3 (logo 3 clusters)
kmeans3 =KMeans(n_clusters=3)
kmeans3.fit(df_flores)

#Armazena o cluster encontrado no k-maens com k igual a 3
df_especies['cluster'] = kmeans3.labels_

#Converte o dataframe df_flores na matriz numpy x utilizando a propriedade values do dataFrame
x = df_flores.values

y = df_especies
#Converte a coluna cluster e a coluna especie do dataframe  df_especies (copiado para y) para os vetores y_kmeans e y_real
y_kmeans = y.cluster.values
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
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'versicolour')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'red', label = 'setosa')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'virginica')

#Plota os centros dos clusters utilziando as caracteristicas de indice 0 (sepal_Length) e indice 1 (sepal_width) na cor amarelo com o marcador diamond (marker = 'D')
plt.scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids', marker = 'D')
plt.legend()
plt.title('Clusterização')
plt.show()
