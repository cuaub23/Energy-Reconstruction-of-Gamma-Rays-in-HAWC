# Procesamiento de Señales de Sensores Masivos y Predicción de Energía mediante Deep Learning (TensorFlow/Keras)

**Problema:** El observatorio HAWC (High-Altitude Water Cherenkov) detecta rayos gamma de muy altas energías. Al entrar en la atmósfera, estos rayos generan cascadas de partículas registradas por una red de tanques de agua. El reto analítico (reconstrucción de energía) consiste en predecir la energía original del rayo gamma a partir de los parámetros indirectos medidos por los sensores.

**Solución:** Diseño, entrenamiento y validación de modelos de Redes Neuronales Profundas (DNN) para mapear las mediciones de los sensores espaciales y temporales hacia un valor continuo de energía original, sustituyendo las costosas simulaciones tradicionales.

**Impacto y Valor:** El modelo final reduce el sesgo (BIAS) a bajas energías (< 1000 GeV) y mantiene un rendimiento altamente competitivo a altas energías frente a los algoritmos estadísticos de reconstrucción actuales. Además, al inferir la energía mediante Deep Learning, se evita la dependencia total de simulaciones de Monte Carlo (`mc.logEnergy`), las cuales representan un costo computacional y temporal restrictivo.

> [!WARNING]
> **Nota sobre los datos:** Los datasets crudos utilizados en este proyecto son de carácter privado y fueron proporcionados exclusivamente por el Instituto de Astronomía (UNAM). Por confidencialidad, no se incluyen en este repositorio.

---

## Arquitectura de Datos y Feature Engineering

El procesamiento de los datos (ETL) y la selección de características (*Feature Engineering*) se documentan en `variables.py` e incluyen las siguientes etapas:

*   **Selección de Variables por Correlación:** Partiendo de la literatura científica del observatorio, se filtraron los parámetros de los sensores midiendo su correlación de Pearson frente a la variable objetivo `mc.logEnergy`. Se conservaron aquellas *features* con correlación $> 0.5$, garantizando entradas con alto poder predictivo para la red neuronal.
*   **Filtros de Dominio (Física):** Se descartaron variables redundantes o con anomalías basándonos en restricciones físicas del experimento.
*   **Transformación Trigonométrica:** Las variables angulares (dirección del rayo) se descompusieron en sus componentes de seno y coseno para eliminar discontinuidades matemáticas y mejorar la convergencia del gradiente durante el entrenamiento.
*   **Pipeline de Entrenamiento (`particiones.py`):** Se implementó una división rigurosa para evitar fuga de datos (*data leakage*):
    *   **20%** reservado como conjunto de Prueba (*Test*) oculto para la evaluación final.
    *   **80%** para Entrenamiento, subdividido internamente en **77.5%** para ajustar pesos y **12.5%** para Validación (monitoreo de métricas por época).

---

## Modelado de Machine Learning (TensorFlow / Keras)

Se diseñaron y evaluaron tres topologías de redes neuronales secuenciales para abordar el problema de regresión. El código de producción del mejor modelo se encuentra encapsulado en `modelo.py`, mientras que la experimentación completa reside en el *notebook*.

**Topología 1 (Modelo Final / Mejor Rendimiento):**
Red densa optimizada para extraer patrones directos sin penalización excesiva.
```python
model_1 = Sequential([
    Dense(64, activation='relu', input_shape=(19,)),
    Dense(64, activation='relu'),
    Dense(1) # Salida de regresión (logEnergy)
])
```
Topología 2 (Control de Sobreajuste con L2):
Inclusión de regularización matemática para penalizar pesos grandes.
```python
model_2 = Sequential([
    Dense(128, activation='relu', input_shape=(19,)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(128, activation='tanh'),
    Dense(1)
])
```
Topología 3 (Regularización Agresiva):
Combinación de L2 y Dropout para forzar a la red a aprender representaciones redundantes y robustas.

```python
model_3 = Sequential([
    Dense(128, activation='relu', input_shape=(19,)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='tanh'),
    Dense(1)
])
```
Resultados y Evaluación del Desempeño

El análisis de error absoluto medio (MAE) y el ajuste gaussiano de los residuales demuestran la viabilidad del Modelo 1 como herramienta de reconstrucción:

[Comparativa de Modelos](grafica.png))

*Altas Energías: La predicción del Modelo 1 supera en precisión (menor dispersión) a los modelos preexistentes del estado del arte.
*Bajas Energías: Aunque el modelo de red neuronal estándar (NN) previo presenta una ligera ventaja en el MAE del ajuste gaussiano en este régimen, nuestro Modelo 1 mantiene un sesgo significativamente menor, logrando un equilibrio robusto en todo el espectro energético.
