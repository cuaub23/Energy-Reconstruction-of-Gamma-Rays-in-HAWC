import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib import colors
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import json 

with open('fctv.json', 'r') as f:
  features_corr_train_val = np.array(json.load(f))

with open('ttv.json', 'r') as f: 
  target_train_val = np.array(json.load(f))

#Escalamiento de los features correlacionados
scaler_corr = StandardScaler()
features_corr_train_val_scaled = scaler_corr.fit_transform(features_corr_train_val)

#Particion del modelo correlacionado
x_train_corr, x_val_corr, y_train_corr, y_val_corr = train_test_split(
    features_corr_train_val_scaled, target_train_val, test_size=0.125, random_state=42)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #Lo usamos para eviatar modelos que empeoran 
checkpoint5 = ModelCheckpoint('mejor_modelo5.h5', monitor='val_loss', save_best_only=True, mode='min')

'''
La red neuronal consta de una entrada de 128 neuronas 2 capas ocultas de 128 donde la segunda incluye un regulalizador l2, la funcion
de activacion cambia ya que esta secuencia demostro tener mejores resultados
'''
model_1_corr = Sequential([
Dense(64, activation='relu', input_shape=(24,)),
Dense(64, activation='relu'),
Dense(1)])

model_1_corr.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history_1_corr = model_1_corr.fit(x_train_corr, y_train_corr,
                    epochs=200,
                    batch_size=512,
                    validation_data=(x_val_corr, y_val_corr),
                    callbacks=[early_stop, checkpoint5])

model_1_corr = load_model('mejor_modelo5.h5')
model_1_corr.save('mejor_modelo5.h5') #Cargamos y guardamos el mejor modelo 
