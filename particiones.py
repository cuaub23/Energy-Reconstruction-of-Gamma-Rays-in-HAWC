import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Caracteristicas sugeridas por el articulo [1] mas caracteristicas agregadas por su correlacion
indices_features_corr = [df.columns.get_loc('rec.zenithAngle'), df.columns.get_loc('rec.coreX'), df.columns.get_loc('rec.coreY'),
          df.columns.get_loc('rec.nHit'), df.columns.get_loc('rec.logNPE'), df.columns.get_loc('rec.logMaxPE'),
          df.columns.get_loc('rec.CxPE40'), df.columns.get_loc('rec.LDFAge'), df.columns.get_loc('rec.PINC'),
          df.columns.get_loc('rec.fAnnulusCharge0'), df.columns.get_loc('rec.fAnnulusCharge1'), df.columns.get_loc('rec.fAnnulusCharge2'),
          df.columns.get_loc('rec.fAnnulusCharge3'), df.columns.get_loc('rec.fAnnulusCharge4'), df.columns.get_loc('rec.fAnnulusCharge5'),
          df.columns.get_loc('rec.fAnnulusCharge6'), df.columns.get_loc('rec.fAnnulusCharge7'), df.columns.get_loc('rec.fAnnulusCharge8'),
          df.columns.get_loc('rec.planeChi2'), df.columns.get_loc('rec.planeNDOF'), df.columns.get_loc('rec.nTankHit'),
          df.columns.get_loc('rec.nTankHitTot'), df.columns.get_loc('rec.nHitTot')]

features_corr = df.iloc[:, indices_features_corr].values #Valores de todas las variables

cos_zenith_fc = np.cos(features_corr[:, 0]).reshape(-1, 1) #Transformamos el angulo 'rec.zenithAngle' a cosenos y senos 
features_corr = np.hstack((features_corr, cos_zenith_fc ))
features_corr[:, 0] = np.sin(features_corr[:, 0])

indice_prediccion = df.columns.get_loc('mc.logEnergy') #Variable a predecir con la red neuronal

all_indices = np.arange(len(df))
#Separamos los indices de df al azar usaremos una muestra del 80% para entrenar y 20% para las pruebas finales 
train_val_indices, final_test_indices = train_test_split(all_indices, test_size=0.2, random_state=42) 

#Valores de entrada y a predecir para la prueba final
final_features_corr_test = features_corr[final_test_indices]
final_target_test = prediccion[final_test_indices]

#Datos a dividir en entrenamiento y validacion para el modelo
features_corr_train_val = features_corr[train_val_indices]
target_train_val = prediccion[train_val_indices]

#Ponemos los datos en archivos JSON
with open('ffct.json', 'w') as f:
    json.dump(final_features_corr_test.tolist(), f)

with open('ftt.json', 'w') as f: 
  json.dump(final_target_test.tolist(), f)

with open('fctv.json', 'w') as f:
  json.dump(features_corr_train_val.tolist(),f)

with open('ttv.json', 'w') as f: 
  json.dump(target_train_val.tolist(), f)
