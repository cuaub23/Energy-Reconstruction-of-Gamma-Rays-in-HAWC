# Procesamiento de Señales de Sensores Masivos y Predicción de Energía mediante Deep Learning (TensorFlow/Keras)

Problema: El observatorio HAWC (High-Altitude Water Cherenkov) es un instrumento que detecta, entre otras cosas, rayos gamma de muy altas energías. Los rayos gamma, al entrar en contacto con la atmósfera, generan cascadas de partículas que HAWC detecta en sus tanques. Un reto importante al que nos enfrentamos es determinar la energía original a partir de los parámetros que HAWC puede medir; este problema es conocido como reconstrucción de energía.

Solucion: Desarrollo y entranamiento de un modelo de red neuronal para la reconstruccion de energia de los rayos gamma detectados por HAWC.

Impacto: El modelo construido es mejor (tiene menor BIAS) a bajas energías (hasta alrededor de 1000 GeV) y equiparable a altas energías respecto a los modelos de reconstruccion estadisticos y de redes neuronales. 

Los valores de energia `mc.logEnergy` se obtuvieron a traves de simulaciones Monte-Carlo. Sin embargo estas son muy costosas computacionalmente por lo que se busca esta alternativa. 
#Arquitectura de los datos 
El articulo numero 3 del notebook indica ciertas variables estan correlacionadas con `mc.logEnergy` se uso esto como base. Posteriormente, se busco a partir de los parametros restantes las variables que tuvieran una correlacion con `mc.logEnergy` mayor a 0.5 y se agregaron en la lista de variables para entrenar a la red neuronal. Esto ya que variables con este tipo de correlacion son mas eficientes para entrenar redes neuronales ya que este numero indica que tanto x podria ser funcion de z que es lo que en principio hace una red neuronal le das x y te regresa y a partir de determinar pesos. De estas ademas se descartaron algunos casos por razones meramente fisicas. 

Se tranforman las variables angulares en senos y cosenos. Ambas para tener un mayor muestreo de variables. 

Todo esto se encuentra dentro del archivo `variables.py`

> [!WARNING]
>Los datos no se encuentran en el repositorio ya que dichos son privados y fueron proporcionados por el Instituto de Astronomia (UNAM) para el desarrollo de este proyecto.

Se desarrollaron 3 modelos principales 
```yaml
#Topologia 1
model_1 = Sequential([
Dense(64, activation='relu', input_shape=(19,)),
Dense(64, activation='relu'),
Dense(1)]) 
```

```yaml
#Topologia 2
model_2 = Sequential([
Dense(128, activation='relu', input_shape=(19,)),
Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
Dense(128, activation='tanh'),
Dense(1)])
```

```yaml
#Topologia 3
model_3 = Sequential([
Dense(128, activation='relu', input_shape=(19,)),
Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
Dropout(0.5),
Dense(128, activation='tanh'),
Dense(1)])
```

Siendo las primeras dos las que obtuvieron los mejores resultados y a las que se les complementento con los columnas de datos extra. 
