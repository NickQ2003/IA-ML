import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['StudyHours', 'PrevExamScore']]  # Features
y = df['Pass']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LASSO model with alpha (regularization parameter)
lasso_model = Lasso(alpha=0.1)

# Train the model on the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lasso_model.predict(X_test)

# Evaluate the model's performance using R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared score: {r2}')

# Display the coefficients of the features
print(f'LASSO Coefficients: {lasso_model.coef_}')

# Try different alpha values and compare the results
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'Alpha: {alpha}, R-squared score: {r2}, Coefficients: {lasso_model.coef_}')

    
# ============================================================
# Clase practica: seleccion de variables con Lasso
# ============================================================
# Objetivo de esta tecnica:
# Usar regularizacion L1 para entrenar un modelo que, ademas de predecir,
# puede reducir algunos coeficientes exactamente a cero.
#
# Idea central:
# Lasso castiga la complejidad del modelo. Si una variable aporta poco,
# su coeficiente puede volverse 0. Cuando eso pasa, podemos interpretarla como
# una variable no seleccionada por el modelo.
#
# Esta tecnica responde la pregunta:
# "Que variables parecen suficientemente utiles como para conservar un
# coeficiente distinto de cero?"

# ============================================================
# 1. Importacion de librerias
# ============================================================
# pandas organiza los datos.
# train_test_split separa datos para entrenamiento y prueba.
# StandardScaler estandariza variables para que Lasso penalice de forma justa.
# Lasso es el modelo de regresion con regularizacion L1.
# r2_score mide rendimiento en regresion.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# ============================================================
# 2. Dataset de ejemplo
# ============================================================
# Cada fila representa un estudiante.
# Usaremos StudyHours y Pass para intentar explicar PrevExamScore.
# Se usa regresion porque Lasso de sklearn predice valores numericos continuos.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Convertimos los datos a DataFrame para trabajar con nombres de columnas.
df = pd.DataFrame(data)

# ============================================================
# 3. Separacion entre variables predictoras y objetivo
# ============================================================
# X contiene variables candidatas. Lasso decidira cuanto peso darles.
X = df[['StudyHours', 'Pass']]

# y es el valor numerico que queremos predecir.
y = df['PrevExamScore']

# ============================================================
# 4. Division en entrenamiento y prueba
# ============================================================
# Entrenamos con una parte y evaluamos con datos separados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 5. Escalado de variables
# ============================================================
# Lasso penaliza coeficientes. Si las variables tienen escalas muy distintas,
# la penalizacion puede ser injusta.
# StandardScaler transforma cada variable para que tenga media 0 y desviacion 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. Entrenamiento del modelo Lasso
# ============================================================
# alpha controla la fuerza de la penalizacion.
# - alpha mas alto: mas castigo, mas coeficientes pueden ir a cero.
# - alpha mas bajo: menos castigo, el modelo se parece mas a regresion lineal.
model = Lasso(alpha=0.1, random_state=42)

# fit entrena el modelo y aprende los coeficientes regularizados.
model.fit(X_train_scaled, y_train)

# ============================================================
# 7. Prediccion y evaluacion
# ============================================================
# predict genera valores numericos estimados para el conjunto de prueba.
y_pred = model.predict(X_test_scaled)

# R2 mide que proporcion de la variacion de y explica el modelo.
r2 = r2_score(y_test, y_pred)

print("Predicted scores:", y_pred)
print("Actual scores:", y_test.values)
print(f"R-squared: {r2}")

# ============================================================
# 8. Interpretacion de coeficientes
# ============================================================
# Los coeficientes indican el peso de cada variable despues de la regularizacion.
# Si un coeficiente es 0, Lasso elimino efectivamente esa variable.
coefficients = pd.Series(model.coef_, index=X.columns)

print("Lasso coefficients:")
print(coefficients)

selected_features = coefficients[coefficients != 0].index.tolist()
print("Selected features using Lasso:", selected_features)

# ============================================================
# 9. Como reflexionar sobre esta tecnica
# ============================================================
# - Ventaja: selecciona variables y controla complejidad al mismo tiempo.
# - Riesgo: si alpha es demasiado alto, puede eliminar variables utiles.
# - Cuidado: conviene escalar variables antes de aplicar Lasso.
# - Pregunta practica: que pasa si cambio alpha? Se mantienen las mismas
#   variables o cambia la seleccion?
