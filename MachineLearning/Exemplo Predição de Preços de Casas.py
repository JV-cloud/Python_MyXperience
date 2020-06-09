
# coding: utf-8


from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import set_option
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

#Defining Working Directory with OS
import os
path=os.chdir("C:/Users/BRJUVEN1/iCloudDrive/MBA_Pós/FIAP/IA e Machine Learning/08 Modelos de IA e ML/Notas de Aula")

# # Load dataset


filename = ("Datasets\housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names,engine='python')

#
#- CRIM     per capita crime rate by town
#- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#- INDUS    proportion of non-retail business acres per town
#- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#- NOX      nitric oxides concentration (parts per 10 million)
#- RM       average number of rooms per dwelling -  é o número médio de cômodos entre os imóveis na vizinhança.
#- AGE      proportion of owner-occupied units built prior to 1940
#- DIS      weighted distances to five Boston employment centres
#- RAD      index of accessibility to radial highways
#- TAX      full-value property-tax rate per $10,000
#- PTRATIO  pupil-teacher ratio by town - é a razão de estudantes para professores nas escolas de ensino fundamental e médio na vizinhança.
#- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#- LSTAT    % lower status of the population - é a porcentagem de proprietários na vizinhança considerados de "classe baixa" (proletariado).
#- MEDV     Median value of owner-occupied homes in $1000's


# Descriptive statistics
# shape
print(dataset.shape)

# types
dataset.dtypes


# head
dataset.head(20)

# descriptions, change precision to 2 places
set_option('precision', 1)
dataset.describe()

# correlation
set_option('precision', 2)
corr = dataset.corr(method='pearson')
sb.heatmap(corr)  



# histograms
dataset.hist(bins=10,figsize=(9,7),grid=False);

# I do feel features 'RM', 'LSTAT', 'PTRATIO', and 'MEDV' are essential. The remaining non-relevant features have been excluded.
# Analysis from above  data:
# 1)increase in RM value increases MEDV value ie price of the home.
# 2) Lower the value of LSTAT higher the value of MEDV
# 3) PTRATIO decrease in the value increases MEDV

prices = dataset['MEDV']
dataset = dataset.drop(['CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'], axis = 1)
features = dataset.drop('MEDV', axis = 1)
dataset.head()

# Split-out validation dataset
seed = 7
test_size = 0.30

X = features.values
Y = prices.values


seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('RD', Pipeline([('Scaler', StandardScaler()),('RD', Ridge())])))

pipelines.append(('LS', Pipeline([('Scaler', StandardScaler()),('LS', Lasso())])))

pipelines.append(('EL', Pipeline([('Scaler', StandardScaler()),('EL', ElasticNet())])))

pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('DTR', Pipeline([('Scaler', StandardScaler()),('DTR', DecisionTreeRegressor())])))

pipelines.append(('RF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))

pipelines.append(('ADA', Pipeline([('Scaler', StandardScaler()),('ADA', AdaBoostRegressor())])))

pipelines.append(('SVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
pipelines.append(('SVR-RBF', Pipeline([('Scaler', StandardScaler()),('SVR', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))])))
pipelines.append(('SVR-Linear', Pipeline([('Scaler', StandardScaler()),('SVR', SVR(kernel='linear', C=100, gamma='auto'))])))
pipelines.append(('SVR-Poly', Pipeline([('Scaler', StandardScaler()),('SVR', SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1))])))
results = []
names = []


# Evaluate Algorithm
# Test options and evaluation metric using Root Mean Square error method
num_folds = 10
RMS = 'neg_mean_squared_error'

for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=RMS)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Comparando os Algorítimos
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1)
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)
svr_rbf.fit(rescaledX, y_train)


rescaledTestX = scaler.transform(X_test)

predictions = svr_rbf.predict(rescaledTestX)

print(metrics.median_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))


predictions=predictions.astype(int)
finalresult = pd.DataFrame({
        "Preço Original": y_test,
        "Preço Predito": predictions
    })

finalresult.to_csv("PredictedPrice.csv", index=False)

filenameresult=("PredictedPrice.csv")
datasetresult = read_csv(filename, names=names,engine='python')
datasetresult.head()
