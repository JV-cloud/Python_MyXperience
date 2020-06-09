from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

# Importe as bibliotecas de Pipelines e Pré-processadores
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Importa o pacote OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# # Base de dados de músicas do Spotify
# https://www.kaggle.com/geomack/spotifyclassification
# https://developer.spotify.com/web-api/get-audio-features/
dataset = pd.read_csv('Datasets\DadosSpotify.csv', sep=',', engine='python')


#Faz a validação via crossvalidation
def Acuracia(clf,X,y):
    resultados = cross_val_predict(clf, X, y, cv=5)
    return metrics.accuracy_score(y,resultados)


# # Pre-processamento de dados
def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0


# Remove features
remove_features(['id','song_title'])


# # Separa a classe dos dados
classes = dataset['target']
dataset.drop('target', axis=1, inplace=True)


# # Label Encoder
enc = LabelEncoder()
inteiros = enc.fit_transform(dataset['artist'])


# Cria uma nova coluna chamada 'artist_inteiros'
dataset['artist_inteiros'] = inteiros

remove_features(['artist'])


# # Pandas Get_dummies
#http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.get_dummies.html
#dataset = pd.get_dummies(dataset, columns=['artist'], prefix=['artist'])

dataset.columns

len(dataset.columns)

# Visualizando as colunas
# pandas object type https://stackoverflow.com/questions/21018654/strings-in-a-dataframe-but-dtype-is-object
dataset.dtypes

# checando missing values
dataset.isnull().sum()


# coluna artistInteiros
dataset.values[:][:,13]


# Instancia um objeto do tipo OnehotEncoder
ohe = OneHotEncoder()

# Transforma em arrayn numpy o dataset.
dataset_array = dataset.values

# Pega o numero de linhas.
num_rows = dataset_array.shape[0]


# Transforma a matriz em uma dimensão
inteiros = inteiros.reshape(len(inteiros),1)

# Criar as novas features a partir da matriz de presença
novas_features = ohe.fit_transform(inteiros)

# Imprime as novas features
novas_features

aux = novas_features.toarray()

# Concatena as novas features ao array
dataset_array = np.concatenate([dataset_array, novas_features.toarray()], axis=1)

# Visualizando a quantidade de linhas e colunas da base
dataset_array.shape


# Transforma em dataframe e visualiza as colunas
dataf = pd.DataFrame(dataset_array)

dataf.head()

#Calculo do MinMaxScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min

#Calculo do StandardScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#z = (x - u) / s - where u is the mean of the training samples and and s is the standard deviation of the training samples

pip_1 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC())
])

pip_2 = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('clf', svm.SVC())
])

pip_3 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='rbf'))
])

pip_4 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='poly'))
])

pip_5 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='linear'))
])

    
# # Teste com apenas labelEncoder nos dados

# Teste com apenas LabelEncoder na coluna 'artist' usando o pipeline 'pip_1'
Acuracia(pip_1,dataset,classes)

Acuracia(pip_2,dataset,classes)
# # Testando Kernels 
# Kernel rbf
Acuracia(pip_3,dataset,classes)

# Kernel Polynomial
Acuracia(pip_4,dataset,classes)

# Kernel Linear
Acuracia(pip_5,dataset,classes)


# # GridSearch

from sklearn.model_selection import GridSearchCV

lista_C = [0.001, 0.01, 0.1, 1, 10,100]
lista_gamma = [0.001, 0.01, 0.1, 1, 10, 100]


parametros_grid = dict(clf__C=lista_C, clf__gamma=lista_gamma)

#Faz o tuning dos parametros testando cada combinação utilziando CrossValidation com 5 folds e analisando a acurácia
grid = GridSearchCV(pip_1, parametros_grid, cv=10, scoring='accuracy', verbose = 1)

grid.fit(dataset,classes)

#Imprime os Melhores parâmetros Encontrados
print(grid.best_params_)

#Imprime os Melhores parâmetros Encontrados
print(grid.best_score_)

# # Métricas de Avaliação de Modelos

pip_6 = Pipeline([
('scaler',StandardScaler()),
('clf', svm.SVC(kernel='rbf',C=100,gamma=0.01))
])


resultados = cross_val_predict(pip_6, dataset, classes, cv=10)

print (metrics.classification_report(classes,resultados,target_names=['0','1']))

metrics.accuracy_score(classes,resultados)

