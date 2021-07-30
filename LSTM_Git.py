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

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor

df = pd.read_csv("C:/Users/abidm/Desktop/Projet/deep_try/Sales.csv")


df['Booking date'] = pd.to_datetime(df['Booking date'], format='%m/%d/%Y')
df['Year'] = df['Booking date'].dt.year 
df['Month'] = df['Booking date'].dt.month 

df.loc[df['Industry'].str.contains("Gov"), 'Type'] = 'Public'
df.loc[df['Industry'].str.contains("Prv"), 'Type'] = 'Private'

df_gr=df.groupby(['GPCH', 'Year','Month','Type'],as_index=False)[['Reportables (ppx)']].sum()
df_pub=df_gr.loc[df_gr['Type']=="Public",]
df_priv=df_gr.loc[df_gr['Type']=="Private",]

df_priv.loc[df_priv['Year']==2015,'Month_Absc']=df_priv.loc[df_priv['Year']==2015,'Month']
df_priv.loc[df_priv['Year']==2016,'Month_Absc']=df_priv.loc[df_priv['Year']==2016,'Month']+12
df_priv.loc[df_priv['Year']==2017,'Month_Absc']=df_priv.loc[df_priv['Year']==2017,'Month']+24
df_priv.loc[df_priv['Year']==2018,'Month_Absc']=df_priv.loc[df_priv['Year']==2018,'Month']+36
df_priv.loc[df_priv['Year']==2019,'Month_Absc']=df_priv.loc[df_priv['Year']==2019,'Month']+48
df_priv.loc[df_priv['Year']==2020,'Month_Absc']=df_priv.loc[df_priv['Year']==2020,'Month']+60

df_pub.loc[df_pub['Year']==2015,'Month_Absc']=df_pub.loc[df_pub['Year']==2015,'Month']
df_pub.loc[df_pub['Year']==2016,'Month_Absc']=df_pub.loc[df_pub['Year']==2016,'Month']+12
df_pub.loc[df_pub['Year']==2017,'Month_Absc']=df_pub.loc[df_pub['Year']==2017,'Month']+24
df_pub.loc[df_pub['Year']==2018,'Month_Absc']=df_pub.loc[df_pub['Year']==2018,'Month']+36
df_pub.loc[df_pub['Year']==2019,'Month_Absc']=df_pub.loc[df_pub['Year']==2019,'Month']+48
df_pub.loc[df_pub['Year']==2020,'Month_Absc']=df_pub.loc[df_pub['Year']==2020,'Month']+60

df_ACTH_priv=df_priv.loc[df_priv['GPCH']=="ACTH",]
df_ACTH_priv_train = df_ACTH_priv.loc[(df_ACTH_priv['Year']<=2019),]
df_ACTH_priv_train_np=df_ACTH_priv_train.to_numpy()
plt.plot(df_ACTH_priv_train_np[:,5],df_ACTH_priv_train_np[:,4],marker='o')

#extraction de la colonne de vente qui est "Reportables"
col=df_ACTH_priv_train.filter(['Reportables (ppx)'])
col=col.values
training_data_len=math.ceil(len(col)*.8)  # extraction de train data
scaler=MinMaxScaler(feature_range=(0,1))   #scaling data
scaled_data=scaler.fit_transform(col)
train_data=scaled_data[0:training_data_len,:]

    
#création de train data
x_train=[]
y_train=[]
for i in range(12,training_data_len):
    x_train.append(train_data[i-12:i,0])
    y_train.append(train_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

def create_model(layers_lstm,layers_dense, activation_input):
    
    model = Sequential()
    
    for i, nodes in enumerate(layers_lstm):
        if i==0:
            model.add(LSTM(nodes, activation=activation_input, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        elif i!=len(layers_lstm)-1 :
            model.add(LSTM(nodes, activation=activation_input, return_sequences=True))
        else:
            model.add(LSTM(nodes, activation=activation_input))
            
    for i, nodes in enumerate(layers_dense):
        model.add(Dense(nodes))
        
#     model.add(Dense(1)) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='mse')
    return model
    
model = KerasRegressor(build_fn=create_model, verbose=0)

#layers_lstm = [[50,50], [50, 100], [50, 80, 100],[50, 100, 150],[100, 50],[50,50,50,50,50],[30,20,10],[200,100,50]]
layers_lstm = [(50,50), (50, 100), (100, 50), (200, 100)]
layers_dense = [(10,1),(50,1),(25,1)]
activation_input = ['relu','sigmoid']
#es
es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience=15)
param_grid = dict(layers_lstm=layers_lstm,layers_dense=layers_dense ,activation_input=activation_input, batch_size = [1], epochs=[300])
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train,callbacks=[es])
[grid_result.best_score_,grid_result.best_params_]
model=grid.best_estimator_



#creation de test data  
test_data= scaled_data[training_data_len-12:,:]
x_test=[]
y_test= col[training_data_len:,:]  
for i in range(12,len(test_data)):
    x_test.append(test_data[i-12:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#prediction et rmse
predictions =model.predict(x_test) 
#predictions =grid.predict(x_test) 
predictions=predictions.reshape(-1,1)
predictions=scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse


#preparing the plot
train=col[:training_data_len]
valid=col[training_data_len:]
concat_test=np.concatenate((col[:48,], valid), axis=0)
concat_pred=np.concatenate((col[:48,],predictions ), axis=0)
union=np.c_[concat_test,concat_pred]
union_df=pd. DataFrame(union)
union_df=union_df.reset_index()
union_df["index"]=union_df["index"]+1
union_df = union_df.rename(columns={0: 'Ventes',1: 'Ventes Estimées'})
union_df.plot(x="index", y=['Ventes Estimées', 'Ventes'],marker="x",xlabel="Numéro De Mois",ylabel="Nombre De Ventes")



