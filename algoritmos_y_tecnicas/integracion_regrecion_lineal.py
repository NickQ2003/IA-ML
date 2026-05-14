# ============================================================
# Integracion de regresion lineal
# ============================================================
# Objetivo del archivo:
# Aprender el flujo basico de un modelo supervisado de regresion.
# En regresion, el valor que queremos predecir es numerico continuo;
# aqui se predice el precio de una casa usando sus pies cuadrados.

# ============================================================
# 1. Importacion de librerias
# ============================================================
# numpy ayuda con calculos numericos, pandas organiza datos en tablas,
# scikit-learn aporta el modelo, la division train/test y las metricas,
# matplotlib permite visualizar los resultados.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# ============================================================
# 2. Creacion del dataset
# ============================================================
# Este dataset es pequeno y manual. Cada fila representa una casa.
# SquareFootage es la variable explicativa y Price es el valor objetivo.
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

# Convertimos el diccionario en un DataFrame para trabajar con columnas
# de forma clara, como si fuera una tabla de Excel o SQL.
df = pd.DataFrame(data)

# Revisamos las primeras filas para confirmar que los datos cargaron bien.
print(df.head())

# ============================================================
# 3. Separacion entre variables de entrada y salida
# ============================================================
# X siempre representa las caracteristicas que el modelo usa para aprender.
# Se escribe con doble corchete para mantener formato de tabla 2D.
X = df[['SquareFootage']]

# y representa la respuesta real que queremos predecir.
y = df['Price']

# ============================================================
# 4. Division en entrenamiento y prueba
# ============================================================
# Entrenamiento: datos con los que el modelo aprende la relacion.
# Prueba: datos separados para medir si el modelo generaliza.
# random_state fija la aleatoriedad para obtener resultados reproducibles.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# shape permite verificar cuantas filas y columnas quedaron en cada grupo.
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# ============================================================
# 5. Creacion y entrenamiento del modelo
# ============================================================
# LinearRegression busca una linea que aproxime la relacion entre X e y.
# En terminos simples: Price = intercepto + coeficiente * SquareFootage.
model = LinearRegression()

# fit entrena el modelo: calcula el intercepto y el coeficiente que mejor
# explican los precios usando los datos de entrenamiento.
model.fit(X_train, y_train)

# El intercepto es el valor base cuando la entrada vale 0.
# El coeficiente indica cuanto cambia el precio por cada pie cuadrado extra.
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# ============================================================
# 6. Prediccion
# ============================================================
# predict aplica la formula aprendida a datos nuevos, en este caso X_test.
y_pred = model.predict(X_test)

# Comparamos predicciones contra valores reales para empezar a razonar
# si el modelo esta cerca o lejos de la realidad.
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

# ============================================================
# 7. Evaluacion del modelo
# ============================================================
# MSE mide el error cuadratico promedio: penaliza mucho errores grandes.
# Mientras mas bajo sea, mejor para este tipo de problema.
mse = mean_squared_error(y_test, y_pred)

# R2 indica que proporcion de la variacion del precio explica el modelo.
# Un valor cercano a 1 suele indicar mejor ajuste; puede ser enganoso
# si el dataset es muy pequeno, como en este ejemplo.
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# ============================================================
# 8. Visualizacion
# ============================================================
# Los puntos azules son datos reales de prueba.
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# La linea roja representa las predicciones del modelo para esos puntos.
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Etiquetas para interpretar el grafico sin tener que leer el codigo.
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Muestra la ventana con el grafico.
plt.show()
