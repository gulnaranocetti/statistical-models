# 1. Generar un conjunto de datos sintético con 150 observaciones y 50 variables (X₁, X₂,
# …, X₅₀). Las variables deben tener correlaciones no nulas entre sí. Para lograrlo,
# utilizar una matriz de correlaciones (provista en el EVA) y generar los datos con una
# distribución normal multivariada.

import numpy as np
import pandas as pd

# Cargar la matriz de correlación
cov_matrix = pd.read_csv('mat_cov.csv').values

np.random.seed(42)
n_samples = 150
n_features = 50

mean = np.zeros(n_features)

# Generar datos sintéticos con Distribución Normal Multivariada
X = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)

# Convertir a DataFrame
X_df = pd.DataFrame(X, columns=[f'X{i + 1}' for i in range(n_features)])
X_df.head()

# ----------------------------------------------------------------------------------------


# 2.  Definir una variable objetivo Y como combinación lineal de las 50 variables, donde
# solo 5 serán informativas (tendrán coeficientes no nulos). Agregar un pequeño ruido
# aleatorio gaussiano con media cero y poca varianza (por ejemplo, 0.25).

np.random.seed(42)
n_informativas = 5
n_variables = X.shape[1]

# Índices de las variables informativas
informativas_idx = np.random.choice(n_variables, n_informativas, replace=False)

coeficientes = np.zeros(n_variables)
coeficientes[informativas_idx] = np.random.uniform(1, 5, size=n_informativas)

# Generar Y como combinación lineal + ruido gaussiano
ruido = np.random.normal(loc=0, scale=0.25, size=X.shape[0])
Y = X @ coeficientes + ruido

# ----------------------------------------------------------------------------------------


# 3. Separar los datos en:
# ● Conjunto de entrenamiento: 100 observaciones
# ● Conjunto de test: 50 observaciones

X_train = X[:100]
y_train = Y[:100]

X_test = X[100:]
y_test = Y[100:]

# ----------------------------------------------------------------------------------------


# 4. Entrenar tres modelos distintos utilizando el conjunto de entrenamiento: Regresión
# lineal simple, Regresión Ridge y Regresión Bayesiana. Graficar los coeficientes
# (pesos) estimados para cada una de las 50 variables (como en el laboratorio 4).
# Comparar los resultados y comentar respecto a las variables que sabemos que
# contienen información. Elegir uno de los modelos para continuar, justificando la
# elección

from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
import matplotlib.pyplot as plt

# Regresión Lineal
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)
coef_lr = modelo_lr.coef_

# Regresión Ridge
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)
coef_ridge = modelo_ridge.coef_

# Regresión Bayesiana
modelo_bayesiano = BayesianRidge()
modelo_bayesiano.fit(X_train, y_train)
coef_bayes = modelo_bayesiano.coef_

# Graficar los coeficientes
plt.figure(figsize=(10, 6))
indices = np.arange(len(coef_lr))

plt.plot(indices, coef_lr, label="Regresión Lineal", linestyle='-', marker='o', alpha=0.7)
plt.plot(indices, coef_ridge, label="Regresión Ridge", linestyle='--', marker='s', alpha=0.7)
plt.plot(indices, coef_bayes, label="Regresión Bayesiana", linestyle=':', marker='^', alpha=0.7)

# Sombrear las variables informativas y señalarlas
for idx in informativas_idx:
    plt.text(idx, coef_lr[idx], f'X{idx + 1}', color='green', ha='center', va='bottom', fontsize=8)

plt.xlabel("Índice de variable")
plt.ylabel("Coeficiente estimado")
plt.title("Comparación de coeficientes estimados por modelo")
plt.axhline(0, color='black', linewidth=1)
plt.legend()
plt.grid(True)
plt.show()

# Mostrar coeficientes reales y estimados en las variables informativas
print("Coeficientes reales y estimados en las variables informativas:\n")
for i in informativas_idx:
    print(f"Variable X{i + 1:02d}:")
    print(f"   Verdadero coeficiente: {coeficientes[i]:.3f}")
    print(f"   Estimado (Lineal):     {coef_lr[i]:.3f}")
    print(f"   Estimado (Ridge):      {coef_ridge[i]:.3f}")
    print(f"   Estimado (Bayesiano):  {coef_bayes[i]:.3f}")
    print("-" * 50)

# theta = (np.linalg.inv(X.T @ X)) @ (X.T @ y_train)
# theta = (np.linalg.inv(X_train.T @ X_train)) @ (X_train.T @ y_train)
# # y_pred = X @ theta
# y_pred = X_train @ theta
#
# mse = np.mean((y_train-y_pred)**2)
#
# sigma2 = mse


# Modelo elegido:
# Visualmente, los coeficientes no informativos:
# En el modelo lineal parece haber mucho ruido en los coeficientes.
# En Ridge y Bayesiana, se acercan mucho a 0.

# Solo 5 variables son realmente informativas, es decir, sus coeficientes son diferentes de 0.
# Las otras 45 variables son ruido, por lo que deberían tener valores cercanos a 0.
# Teniendo esto en cuenta, el modelo lineal quedaría descartado.

# El modelo depende de un solo parametro de ajuste, que influye en la intesidad de penalización. Con validación curzada
# se lo puede optimizar hasta encontrar el punto óptimo entre sesgo y varianza.
# En el modelo Bayesiano hay múltiples parámetros de precisión, por lo que requiere de algún método iterativo para
# estimar los coeficientes. Esto hace que el proceso sea más complejo y computacionalmente costoso.

# A partir de los números:
# La regresión lineal produce coeficientes inestables y valores extremadamente alejados de los verdaderos.
# En cambio, tanto Ridge como la Bayesiana logran estimaciones estables y razonablemente cercanas a los valores reales.

# Se concluye que la mejor opción es Ridge.


# ----------------------------------------------------------------------------------------


# # 5. Aplicar un análisis de componentes principales (PCA) al conjunto de entrenamiento.
# Graficar las dos primeras componentes principales (PC1 y PC2) para visualizar la
# estructura de los datos

# Se procede a trabajar con PCA estandarizado sobre el conjunto de entrenamiento, para ello se estandarizan nuestros
# datos ya que PCA es sensible a la escala de los datos (recae en distancias euclidianas).

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Se estandarizan los valores de las variables observadas para el PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# La idea es proyectar sobre las dos componentes principales.

# PCA sobre las dos principales componentes 
pca = PCA(n_components=2)
pca.fit(X_train_scaled)
X_pca = pca.transform(X_train_scaled)
print(X_pca.shape)

# Luego graficamos las observaciones proyectadas:
plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.9)
plt.title("Proyección PCA (entrenamiento)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------------------------


# 6. Dividir el conjunto de test en dos subconjuntos iguales:
# ● A: 25 observaciones
# ● B: 25 observaciones
# Generar un vector aleatorio de tamaño 50 (por ejemplo, con valores uniformes entre
# -10 y 10) y sumarlo a todas las filas del conjunto A. De esta manera, el conjunto A
# representará datos provenientes de una distribución desplazada, mientras que B
# mantendrá la distribución original.

# Particionamos el conjunto de testing en dos subconjuntos: uno con la distribucion original y otro desplazado.

X_test_A = X_test[:25].copy()
X_test_B = X_test[25:].copy()

# Luego generamos un desplazamiento aleatoreo a cada observacion (fila)

shift = np.random.uniform(-10, 10, size=50)
X_test_A_shifted = X_test_A + shift

# ----------------------------------------------------------------------------------------


# 7. Usar el modelo elegido en la parte 2 para predecir sobre ambos conjuntos (A
# modificado y B). Calcular el error de predicción en cada observación. Comentar los
# resultados.

# Para esta seccion, se predice sobre ambos conjuntos de testing la salida correspondiente y se evalua el desempeno del
# modelo con respecto a la metrica de error para ambos conjuntos.

# i. Predecir en ambos subconjuntos
y_pred_A = modelo_ridge.predict(X_test_A_shifted)
y_pred_B = modelo_ridge.predict(X_test_B)

# ii. Calcular errores por observación
errores_A = (y_test[:25] - y_pred_A) ** 2  # error cuadrático observación a observación
errores_B = (y_test[25:] - y_pred_B) ** 2

# iii. Error global promedio para cada subconjunto
mse_A = np.mean(errores_A)
mse_B = np.mean(errores_B)

print("MSE conjunto A (modificado) =", mse_A)
print("MSE conjunto B (original)   =", mse_B)

# Interpretación de Resultados:
# El conjunto B proviene de la misma distribución que los datos de entrenamiento.
# Por lo tanto, el modelo debería predecir bien, o sea tener un error pequeño.

# El conjunto A fue desplazado sumándole una constante random a todas las variables.
# Eso cambia la distribución de entrada, entonces el modelo ya no reconoce estos datos como similares a los entrenados.
# Por lo tanto, el error debería ser más grande.

# Conclusión:
# El modelo aprende una relación válida solo dentro de la distribución de entrenamiento, pero cuando los datos cambian
# ligeramente (distribución desplazada), el desempeño del modelo empeora.


# ----------------------------------------------------------------------------------------


## 8. Proyectar los conjuntos A y B en el espacio de las componentes principales
# obtenidas en la parte 3 y graficar las proyecciones (PC1 vs PC2), coloreando los
# puntos según el error de predicción. Comentar los resultados.

# Escalar ambos conjuntos con el mismo scaler del entrenamiento
X_test_A_scaled = scaler.transform(X_test_A_shifted)
X_test_B_scaled = scaler.transform(X_test_B)

# Proyectar ambos conjuntos en las dos primeras componentes principales
X_test_A_pca = pca.transform(X_test_A_scaled)
X_test_B_pca = pca.transform(X_test_B_scaled)

# con Ridge
# Predicciones del modelo en cada conjunto
y_pred_A = modelo_ridge.predict(X_test_A_shifted)
y_pred_B = modelo_ridge.predict(X_test_B)

# Error absoluto (podrías usar cuadrático también)
error_A = np.abs(y_test[:25] - y_pred_A)
error_B = np.abs(y_test[25:] - y_pred_B)

import matplotlib.pyplot as plt

# i. Estandarizar ambos conjuntos usando el scaler del entrenamiento
X_A_scaled = scaler.transform(X_test_A_shifted)
X_B_scaled = scaler.transform(X_test_B)

# ii. Proyectar en las dos primeras componentes principales
X_A_pca = pca.transform(X_A_scaled)
X_B_pca = pca.transform(X_B_scaled)
X_train_pca = pca.transform(X_train_scaled)

# iii. Definir rango de color común para los errores
vmin = min(errores_A.min(), errores_B.min())
vmax = max(errores_A.max(), errores_B.max())

# iv. Graficar todo en una misma figura
plt.figure(figsize=(9, 7))

# Nube gris: conjunto de entrenamiento
plt.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1],
    color='lightgray', alpha=0.5, s=60,
    label="Entrenamiento"
)

# Conjunto B (original)
plt.scatter(
    X_B_pca[:, 0], X_B_pca[:, 1],
    c=errores_B, cmap="RdYlGn_r",
    vmin=vmin, vmax=vmax,
    label="Conjunto B (original)",
    s=90, marker="o", edgecolor='k', alpha=0.9
)

# Conjunto A (desplazado)
plt.scatter(
    X_A_pca[:, 0], X_A_pca[:, 1],
    c=errores_A, cmap="RdYlGn_r",
    vmin=vmin, vmax=vmax,
    label="Conjunto A (desplazado)",
    s=100, marker="^", edgecolor='k', alpha=0.9
)

# Barra de color compartida
cbar = plt.colorbar()
cbar.set_label("Error cuadrático (por observación)", rotation=270, labelpad=18)

# Configuración del gráfico
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proyección PCA coloreada según error de predicción")
plt.legend(title="Conjunto de datos")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Comentarios
# Los puntos del conjunto B se concentran en una región similar a la nube de entrenamiento.
# Esto confirma que el modelo mantiene un buen desempeño cuando los datos provienen de la misma distribución que la
# usada para el entrenamiento, reproduciendo correctamente la relación entre las variables informativas y la variable
# objetivo.

# Observaciones sobre el conjunto A (modificado)
# Por otro lado, los puntos del conjunto A muestran una mayor dispersión.
# Además, muestran errores significativamente mayores.
# Esto es coherente con el desplazamiento artificial aplicado a las variables de entrada, ya que al modificar la media
# de cada atributo, el modelo Ridge ya no logra generalizar correctamente.

# Conclusiones
# El modelo generaliza bien dentro de la misma distribución.
# Su rendimiento empeora frente a un cambio de distribución.


# ----------------------------------------------------------------------------------------


# 9. Si no conociera las constantes que se le sumaron a las variables del conjunto A,
# ¿qué estrategia elegiría para mejorar las predicciones en ese conjunto?

# Si no se conocieran las constantes sumadas a las variables del conjunto A, la disminución en el desempeño del modelo
# indicaría la presencia de un cambio de distribución.
# Para mejorar las predicciones se podría reestandarizar las variables utilizando las medias y desviaciones del
# conjunto nuevo, de modo de compensar posibles desplazamientos en escala o centro.
