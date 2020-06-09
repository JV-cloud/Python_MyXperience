# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 02:33:32 2019

@author: felip
"""

import pandas as pd
from pandas import read_csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Realiza a leitura do csv contendo uma amostra reduzida dos dados do dataset titanic
dataset = read_csv('https://telescopeinstorage.blob.core.windows.net/datasets/titanic-apriori.csv', sep=';' , engine='python', error_bad_lines=False)
dataset.head()


#Obtêm a quatidade de linhas e colunas
qtdlinhas = dataset.shape[0]
qtdcols = dataset.shape[1]

print(qtdlinhas)
print(qtdcols)

#Converte o dataset em uma lista de transacoes
transacoes = []
for i in range(0, qtdlinhas):
    linhaTransacao = []
    for j in range(0, qtdcols):        
        linhaTransacao.append(str(dataset.values[i,j]))
    
    transacoes.append(linhaTransacao)
print(transacoes)

te = TransactionEncoder()

#Coloca em memórias as trasações e interpreta a quantidade de colunas que serão geradas durante o processamento
te.fit(transacoes)

#O objeto TransactionEncoder faz a conversão das transações em uma matriz binária onde cada linha da matriz representa uma transação
matriz_transacoes = te.transform(transacoes)

#Cria um dataframe auxiliar com a matriz binária (passo te.transform(transacoes)) de transações e as colunas obtidas (passo te.fit(transacoes))
dfAuxiliar = pd.DataFrame(matriz_transacoes, columns=te.columns_)

#Obtêm os itemsets mais frequentes com um suporte mínimo igual a 0.01. O paramêtro use_colnames significa que vamos usar os nomes das colunas do DataFrame dfAuxiliar 
#para construir as regras de Associação
itemsets_freq = apriori(dfAuxiliar, min_support=0.005, use_colnames=True)

#Algumas métricas:
#- support(A->C) = support(A+C) [aka 'support'], range: [0, 1]
#- confidence(A->C) = support(A+C) / support(A), range: [0, 1]
#- lift(A->C) = confidence(A->C) / support(C), range: [0, inf]
#- leverage(A->C) = support(A->C) - support(A)*support(C), range: [-1, 1]
#- conviction = [1 - support(C)] / [1 - confidence(A->C)],

#Obtêm as regras de associação a partir dos itemsets mais frequêntes
regras = association_rules(itemsets_freq, metric="confidence", min_threshold=0.4)

#Ordena as Regras por confiança
regrasOrdenadas = regras.sort_values('confidence' , ascending=False)

#mantém apenas as colunas que vamos utilizar 
regrasOrdenadas = regrasOrdenadas[['antecedents', 'consequents', 'support', 'confidence']]
print(regrasOrdenadas)

#Analise apenas da coluna Survived
regras_sobrevivetes =  regrasOrdenadas[regrasOrdenadas['consequents'] == {'Yes'}]
#OU
subset_sobrevivou = {'Yes'}
regras_sobrevivetes =  regrasOrdenadas[  regrasOrdenadas['consequents'].apply(lambda x: subset_sobrevivou.issubset(x))]

regras_naoSobrevivetes =  regrasOrdenadas[regrasOrdenadas['consequents'] == {'No'}]

subset_Mulheres = {'Female'}
regras_mulheres = regrasOrdenadas[  regrasOrdenadas['antecedents'].apply(lambda x: subset_Mulheres.issubset(x))]
print(regras_mulheres)

#Concatena as regras relacionadas dos sobreviventes e não-sobreviventes para análise única
regrasGeral =  pd.concat([regras_sobrevivetes,regras_naoSobrevivetes])

regrasGeral = regrasGeral.sort_values('confidence' , ascending=False)

print(regrasSovreviventes)
