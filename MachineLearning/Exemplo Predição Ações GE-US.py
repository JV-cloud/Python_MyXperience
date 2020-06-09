import numpy as np
import sys
import time
import pandas as pd 

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


TIME_STEPS =60
BATCH_SIZE = 20
EPOCHS = 20

def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")


def trim_dataset(mat,batch_size):
    """
    Torna o dataset divisivel pelo tamanho do BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def build_timeseries(mat, y_col_index):
    """
    Converte o o dataset mat em um formato de aprendizado supervisionado para o LSTM baeado na quantidade de passos (que vão compor cada amostra de treinamento)
    e o Y sendo a quantidade de passos +1 da coluna que será predita (y_col_index)
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    print("dim_0",dim_0)
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
#         if i < 10:
#           print(i,"-->", x[i,-1,:], y[i])
    print("Tamanho da Série i/o",x.shape,y.shape)
    return x, y


stime = time.time()

#Faz a leitura do dataset - versão reduzida para qque o treinamento possa ser feito em sala de aula
df_ge = pd.read_csv( "ge-short.us.txt", engine='python')
print(df_ge.shape)
print(df_ge.columns)
print(df_ge.dtypes)

#Colunas a serem utilizadas
train_cols = ["Open","High","Low","Close","Volume"]

df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)

print("Train--Test size", len(df_train), len(df_test))

#Normalizaçãpo com MinMaxScalar
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

#Apaga os dataframes não utilizados
del df_ge
del df_test
del df_train
del x

#Controi o conjunto de treinamento baseado na função build_timeseries
x_t, y_t = build_timeseries(x_train, 3)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
print("Tamanho de x_t e y_t",x_t.shape, y_t.shape)


def create_model():
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(60, dropout=0.0)) #, return_sequences=True
    lstm_model.add(Dropout(0.2))
    #lstm_model.add(LSTM(40, dropout=0.0 ))
    #lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    #optimizer = optimizers.RMSprop(lr=params["lr"])
    #optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = optimizers.Adam()
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model


x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)
    




print("Verifica se há GPU disponível", K.tensorflow_backend._get_available_gpus())
model = create_model()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=80, min_delta=0.0001)

mcp = ModelCheckpoint( "best_model.h5", monitor='val_loss', verbose=1,
                      save_best_only=True, save_weights_only=False, mode='min', period=1)


history = model.fit(x_t, y_t, epochs=EPOCHS, verbose=1, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                    trim_dataset(y_val, BATCH_SIZE)), callbacks=[es, mcp])


# model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE
y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Erro ", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])

# convert the predicted value to range of real data
y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
#y_pred_org =  min_max_scaler.inverse_transform(y_pred)
y_test_t_org = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
#y_test_t_org = min_max_scaler.inverse_transform(y_test_t)

print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the training data
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# load the saved best model from above
saved_model = load_model( 'best_model.h5') 
print(saved_model)

y_pred_train = saved_model.predict(trim_dataset(x_t, BATCH_SIZE), batch_size=BATCH_SIZE)

y_pred_train =y_pred_train.flatten()
y_pred_train_plot = (y_pred_train * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
y_train_plot = (y_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

plt.figure()
plt.plot(y_pred_train_plot)
plt.plot(y_train_plot)
plt.title('Prediction vs Real Stock Price - Training')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left') 
plt.show()

y_pred_test = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred_test = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred_test)
print("Error is", error, y_pred_test.shape, y_test_t.shape)
print(y_pred_test[0:15])
print(y_test_t[0:15])


y_pred_test_plot = (y_pred_test * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
y_test_plot = (y_test_t * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]

print(y_pred_test_plot[0:15])
print(y_test_plot[0:15])

# Visualize the prediction
plt.figure()
plt.plot(y_pred_test_plot)
plt.plot(y_test_plot)
plt.title('Prediction vs Real Stock Price - Test')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
#plt.show()

print_time("Término ", stime)