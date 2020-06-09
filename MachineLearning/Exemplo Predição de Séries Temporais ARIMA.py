# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:55:00 2019

@author: felip
"""

import pandas as pd
import matplotlib.pyplot as plt
#pip install statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
#pip install pyramid.arima
from pmdarima  import auto_arima


#https://fred.stlouisfed.org/series/IPG2211A2N
data = pd.read_csv('d:\Temp\Electric_Production.csv',index_col=0)
data.head()

data.index = pd.to_datetime(data.index)
data.head()

data.columns = ['Energy Production']

title="Produção de Energia Jan 1985--Maio 2019"
plt.title(title)
plt.plot(data.index, data)

rcParams['figure.figsize'] = 18, 8
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plt.show()


train = data[:int(0.75*(len(data)))]
test = data[int(0.75*(len(data))):]

#Akaike information criterion (AIC) - estimador estatistico da qualidade da predição, quanto melhor melhor
#Execução paralela
full_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, n_jobs = 8, stepwise = False)

#Faz um GridSearch procurando a melhor combinação de parâmetros p, q e d, m é o tamanho do ciclo sazonal, no nosso caso 12 meses


# stepwise_model = auto_arima(train, start_p=1, start_q=1,
#                            max_p=3, max_q=3, m=12,
#                            start_P=0, seasonal=True,
#                            d=1, D=1, trace=True,
#                            error_action='ignore',  
#                            suppress_warnings=True, 
#                            stepwise=True)




#plotting the data
train['Energy Production'].plot()
test['Energy Production'].plot()


# AIC do modelo encontrado (quanto menor melhor)
# Akaike Information Criterion:
# -2 * llf + 2 * df_model
# onde df_model representam os parâmetros p e q  e llf a é o log-verossimilhança dos dados
print(full_model.aic())

#Configuração do modelo encontrado
#(2, 1, 1, 12)
print(full_model.seasonal_order)

forecast = full_model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])


plt.plot(data, label='Série Original')
#plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

#Diminuindo o conjunto de treinamento para observar apenas os ultimos anos  anos e prever 2018 até 2019

train = data.loc[pd.Timestamp('2010-01-01'):pd.Timestamp('2017-12-30')]
test =data.loc[pd.Timestamp('2018-01-01'):]

stepwise_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True)


# AIC do modelo encontrado (quanto menor melhor)
# Akaike Information Criterion:
# -2 * llf + 2 * df_model
# Where df_model (the number of degrees of freedom in the model)
print(stepwise_model.aic())

#Configuração do modelo encontrado
#(2, 1, 1, 12)
print(stepwise_model.seasonal_order)


forecast = stepwise_model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])


data2010_2019 = data.loc[pd.Timestamp('2010-01-01'):]
plt.plot(data2010_2019, label='Train')
#plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

#Prevendo os próximos 24 períodos
forecast24 = stepwise_model.predict(n_periods=24)
idx = pd.date_range('2018-01-01', periods=24, freq='MS')

forecast24 = pd.DataFrame(forecast24,columns=['Prediction'])
forecast24.index = pd.to_datetime(idx)

plt.plot(data, label='Train')
plt.plot(forecast24, label='Prediction 24')
plt.show()

