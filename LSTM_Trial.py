#import modules
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import math
from sklearn.preprocessing import MinMaxScaler
#pip install --upgrade numpy
#import tensorflow
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor 

df_priv= pd.read_csv("C:/Users/user/Desktop/Projects/Prediction_ventes/deep_try/Our_Sales_Prv.csv")

df_priv.loc[df_priv['Year']==2015,'Month_Absc']=df_priv.loc[df_priv['Year']==2015,'Month']
df_priv.loc[df_priv['Year']==2016,'Month_Absc']=df_priv.loc[df_priv['Year']==2016,'Month']+12
df_priv.loc[df_priv['Year']==2017,'Month_Absc']=df_priv.loc[df_priv['Year']==2017,'Month']+24
df_priv.loc[df_priv['Year']==2018,'Month_Absc']=df_priv.loc[df_priv['Year']==2018,'Month']+36
df_priv.loc[df_priv['Year']==2019,'Month_Absc']=df_priv.loc[df_priv['Year']==2019,'Month']+48
df_priv.loc[df_priv['Year']==2020,'Month_Absc']=df_priv.loc[df_priv['Year']==2020,'Month']+60

df_ACTH_priv=df_priv.loc[df_priv['GPCH']=="ACTH",]

df_ACTH_priv_train = df_ACTH_priv.loc[(df_ACTH_priv['Year']<=2019),]
df_ACTH_priv_train_np=df_ACTH_priv_train.to_numpy()
plt.plot(df_ACTH_priv_train_np[:,5],df_ACTH_priv_train_np[:,4],marker='o')

col=df_ACTH_priv_train.filter(['Reportables (ppx)'])
col=col.values
training_data_len=math.ceil(len(col)*.8)  # extraction de train data
scaler=MinMaxScaler(feature_range=(0,1))   #scaling data
scaled_data=scaler.fit_transform(col)
train_data=scaled_data[0:training_data_len,:]

x_train=[]
y_train=[]
for i in range(12,training_data_len):
    x_train.append(train_data[i-12:i,0])
    y_train.append(train_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
x_seen=x_train
y_seen=y_train
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
y_train.shape

def create_model():
    model = Sequential()
    #model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    #model.add(LSTM(12, activation='relu', return_sequences=True,input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(32, activation='relu',input_shape=(x_train.shape[1], 1),return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    #model.add(Dense(10))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model
model = KerasRegressor(build_fn=create_model, verbose=0)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50)
history=model.fit(x_train, y_train,batch_size=36, epochs=400, verbose=1, validation_split=0.1,callbacks=[es])

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#creation de test data  
test_data= scaled_data[training_data_len-12:,:]
x_test=[]
y_test= col[training_data_len:,:]  
for i in range(12,len(test_data)):
    #x_test.append(train_data[i-12:i,0])
    x_test.append(test_data[i-12:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape
#prediction et rmse
predictions =model.predict(x_test) 
#predictions =grid.predict(x_test) 
predictions=predictions.reshape(-1,1)
predictions=scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse_scaled=scaler.transform(rmse.reshape(1, -1))
print("rmse=",rmse,"and","rmse_scaled=",rmse_scaled)

#preparing the plot
train=col[:training_data_len]
valid=col[training_data_len:]
concat_test=np.concatenate((col[:training_data_len,], valid), axis=0)
concat_pred=np.concatenate((col[:training_data_len,],predictions ), axis=0)
union=np.c_[concat_test,concat_pred]
union_df=pd. DataFrame(union)
union_df=union_df.reset_index()
union_df["index"]=union_df["index"]+1
union_df = union_df.rename(columns={0: 'Ventes',1: 'Ventes Estimées'})
union_df.plot(x="index", y=['Ventes Estimées', 'Ventes'],marker="x",xlabel="Numéro De Mois",ylabel="Nombre De Ventes")


