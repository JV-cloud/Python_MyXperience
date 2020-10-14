
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



import os
os.environ['PATH'].split(os.pathsep)
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


# A primeira coisa que vamos fazer é ler o conjunto de dados usando a função read_csv() dos Pandas. 
# Colocaremos esses dados em um DataFrame do Pandas, chamado "titanic", e nomearemos cada uma das colunas.


url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic.head()


# VARIABLE DESCRIPTIONS
# Survived - Survival (0 = No; 1 = Yes);
# Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd);
# Name - Name;
# Sex - Sex;
# Age - Age;
# SibSp - Number of Siblings/Spouses Aboard;
# Parch - Number of Parents/Children Aboard;
# Ticket - Ticket Number;
# Fare - Passenger Fare (British pound);
# Cabin - Cabin;
# Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton);

# Como estamos construindo um modelo para prever a sobrevivência de passageiros do Titanic, nosso alvo será a variável "Survived" do dataframe titanic.
# Para ter certeza de que é uma variável binária, vamos usar a função countplot () do Seaborn.


sb.countplot(x='Survived',data=titanic, palette='hls')


# Ok, agora veja que a variavel Survived é binária

# # Checking for missing values
# É fácil checar missing values usando método isnull() com o método sum(), o número retornado condiz com a quantidade True para o teste, ou seja, quantidade de valores nulos nas variaveis

titanic.isnull().sum()

titanic.info()


# Ok, então existem 891 linhas no dataframe. Cabin é quase todo composto por missing values, então podemos eliminar essa variável completamente! Mas e quanto à idade? A age parece um preditor relevante para a sobrevivência, certo? Nós queremos manter as variáveis, mas tem 177 missing values. Precisamos encontrar uma maneira de nos aproximarmos desses valores em falta!

# # Lidando com missing values
# Removendo missing values
# Vamos além dos missing values... Vamos descartar todas as variáveis que não são relevantes para a predição de Survived.


#E quanto ao nome de uma pessoa, número do bilhete e número de identificação do passageiro? Eles são irrelavantes para prever a capacidade de sobrevivência. E, como você se lembra, a variável cabine é quase todoa com missing values!!
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)
titanic_data.head()


# Agora, o dataframe foi reduzido para apenas variáveis relevantes, mas agora precisamos lidar com os valores ausentes na variável age.

# # Imputing missing values
# Vejamos como a idade do passageiro está relacionada à sua classe como passageiro no barco.


sb.boxplot(x='Pclass', y='Age', data=titanic_data, palette='hls')


# Falando mais ou menos, poderíamos dizer que quanto mais jovem é um passageiro, mais provável é que ele esteja na 3ª classe. Quanto mais velho for um passageiro, maior a probabilidade de estarem na 1ª classe. Portanto, há um relacionamento frouxo entre essas variáveis. Então, vamos escrever uma função que se aproxime da idade dos passageiros, com base em sua classe. Na caixa, parece que a idade média dos passageiros de 1ª classe é de cerca de 37 anos, os passageiros de 2ª classe são 29 e os passageiros de 3ª classe são 24.
# 
# Então, vamos escrever uma função que encontre cada valor nulo na variável Age e, para cada nulo, verifique o valor da Pclass e atribua um valor de idade de acordo com a idade média dos passageiros nessa classe.


def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# Após definir a função, vamos executar apenas para os valores nulos de Age

titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)

titanic_data.isnull().sum()
# Existem ainda 2 valores nulos na variável Embarked. Podemos eliminar esses dois registros sem perder muitas informações importantes do nosso conjunto de dados, então faremos isso.


titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()

# # Converting categorical variables to a dummy indicators

gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
gender.head()

embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark_location.head()

#Ainda não alteramos os dados! Vamos alterar em seguida.
titanic_data.head()

titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)
titanic_data.head()

titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)
titanic_dmy.head()

# Agora temos um conjunto de dados com todas as variáveis no formato correto!

# # Validando independencia entre as variáveis
sb.heatmap(titanic_dmy.corr())  


# Fare e Pclass não sao independentes uma com a outra, entao vou excluí-las.

titanic_dmy.drop(['Pclass'],axis=1,inplace=True)
titanic_dmy.head()

titanic_dmy.drop(['Q'],axis=1,inplace=True)
titanic_dmy.head()


# # Agora vamos lá!!
# 1º: Separar o conjunto em variavel resposta e variaveis de treinamento

X = titanic_dmy.iloc[:,[1,2,3,4,5,6]].values
y = titanic_dmy.iloc[:,0].values

# Agora dividir em treino e teste (teste com 30%)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

#classificador Naive Bayes Gaussiano
classificador = GaussianNB()

classificador.fit(X_train, y_train)

#Em caso de datasets muitos granges é possível utilizar a função partial_fit
#classificador.partial_fit(X_train, y_train)
 
y_pred = classificador.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))





