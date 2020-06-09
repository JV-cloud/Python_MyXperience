# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:36:29 2019

@author: felip
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA


#This case requires to develop a customer segmentation to define marketing strategy. The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.
#Following is the Data Dictionary for Credit Card dataset :

#CUST_ID : Identification of Credit Card holder (Categorical) 
#BALANCE : Balance amount left in their account to make purchases  - Valor do saldo restante em sua conta para fazer compras 
#BALANCE_FREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated) - Com que frequência o saldo é atualizado, pontuação entre 0 e 1 (1 = atualizado com frequência, 0 = não atualizado com frequência)
#PURCHASES : Amount of purchases made from account ONEOFF_PURCHASES : Maximum purchase amount done in one-go 
#INSTALLMENTS_PURCHASES : Amount of purchase done in installment  - Valor da compra parcelado
#CASH_ADVANCE : Cash in advance given by the user - Dinheiro a vista dado pelo usuário
#PURCHASES_FREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased) 
#ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased) - Com que frequência as compras acontecem de uma só vez (1 = comprado com frequência, 0 = comprado com pouca frequência)
#PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done) - Com que frequência as compras parceladas estão sendo feitas
#CASHADVANCEFREQUENCY : How frequently the cash in advance being paid CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
#PURCHASES_TRX : Numbem of purchase transactions made 
#CREDIT_LIMIT : Limit of Credit Card for user 
#PAYMENTS : Amount of Payment done by user 
#MINIMUM_PAYMENTS : Minimum amount of payments made by user 
#PRCFULLPAYMENT : Percent of full payment paid by user 
#TENURE : Tenure of credit card service for user

#Faz a leuitura do dataset
dataset = pd.read_csv('Datasets/CC GENERAL.csv')
dataset = dataset.drop('CUST_ID', axis = 1) 

#Preenche as células vazias (NA) copiando a próxima observação válida para esta que é NA
dataset.fillna(method ='ffill', inplace = True) 

dataset.head()

#Normaliza os Dados:
scaler = StandardScaler() 
dataset_norm = scaler.fit_transform(dataset) 


#TODO - procurar o melhor valor de k para a fazer a Clusterização com o Método Elbow, execute para um k entre 1 e 50
scores = []
inertias =[]
for i in tqdm( range(2, 25)):
    kmeans = KMeans(n_clusters = i, max_iter=100).fit(dataset_norm)    
    #soma dos quadrados intra-clusters de cada cluster
    inertias.append(kmeans.inertia_)
    scores.append( silhouette_score(dataset.values, kmeans.labels_))


    
plt.figure(1)
plt.plot(range(2, 25), inertias)
plt.title('O Metodo Elbow')
plt.xlabel('No de clusters')
plt.ylabel('WSS - within cluster sum of squares')
plt.show()

plt.figure(2)
plt.bar(range(2, 25), scores,  align='center', alpha=0.5)
plt.title('O Metodo Silhouette Score')
plt.xlabel('No de clusters')
plt.ylabel('score')
plt.show()


bestKmeans = KMeans(n_clusters = 3, max_iter=100).fit(dataset_norm)

clusters=pd.concat([dataset, pd.DataFrame({'cluster':bestKmeans.labels_})], axis=1)
clusters.head()

for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)

labels =bestKmeans.labels_ 

#Como plotar graficamentes estes Clusters?
    
pca = PCA(n_components=2).fit_transform(dataset_norm)

plt.figure(3)
#Faz a plotagem dos registros do primeiro cluster 0 (y_kmeans == 0) e utiliza as colunas de indice 0 e 1 (sepal_Length e sepal_width) para a plotagem dos pontos nas intruções x[y_kmeans == 0, 0] e x[y_kmeans == 0, 1]
plt.scatter(pca[labels == 0, 0], pca[labels == 0, 1], s = 2, c = 'blue', label = 'Cluster 0')
plt.scatter(pca[labels == 1, 0], pca[labels == 1, 1], s = 2, c = 'red', label = 'Cluster 1')
plt.scatter(pca[labels == 2, 0], pca[labels == 2, 1], s = 2, c = 'green', label = 'Cluster 2')


plt.legend()
plt.title('Clusterização')
plt.show()

    
    