# ============================================================
# Aplicacion de regresion logistica
# ============================================================
# Objetivo del archivo:
# Aprender un modelo supervisado de clasificacion binaria.
# A diferencia de la regresion lineal, aqui no se predice un numero continuo,
# sino una clase: reprobar (0) o aprobar (1).

# ============================================================
# 1. Importacion de librerias
# ============================================================
# numpy genera rangos numericos para graficar, pandas maneja tablas,
# scikit-learn aporta el modelo, la division de datos y las metricas,
# matplotlib permite ver la curva de probabilidad.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ============================================================
# 2. Creacion del dataset
# ============================================================
# Cada fila representa un estudiante.
# StudyHours es la entrada y Pass es la etiqueta real: 0 = reprueba, 1 = aprueba.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Convertimos los datos a DataFrame para seleccionarlos por columnas.
df = pd.DataFrame(data)

# Vista rapida para validar estructura y valores iniciales.
print(df.head())

# ============================================================
# 3. Separacion entre variables de entrada y salida
# ============================================================
# X contiene lo que el modelo observa: horas de estudio.
X = df[['StudyHours']]

# y contiene la clase correcta que el modelo debe aprender a predecir.
y = df['Pass']

# ============================================================
# 4. Division en entrenamiento y prueba
# ============================================================
# El modelo aprende con X_train/y_train y luego se evalua con X_test/y_test.
# Esto evita evaluar solo con los mismos datos que uso para aprender.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Revisamos tamanos para entender cuantas muestras hay en cada conjunto.
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# ============================================================
# 5. Creacion y entrenamiento del modelo
# ============================================================
# LogisticRegression aprende una frontera de decision usando probabilidades.
# Internamente usa una funcion sigmoide para convertir el resultado en una
# probabilidad entre 0 y 1.
model = LogisticRegression()

# fit ajusta los parametros del modelo a partir de los ejemplos conocidos.
model.fit(X_train, y_train)

# El intercepto y coeficiente definen la curva logistica aprendida.
# Si el coeficiente es positivo, mas horas aumentan la probabilidad de aprobar.
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# ============================================================
# 6. Prediccion
# ============================================================
# predict devuelve la clase final, no la probabilidad: 0 o 1.
y_pred = model.predict(X_test)

# Comparamos lo que predijo el modelo contra las respuestas reales.
print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)

# ============================================================
# 7. Evaluacion del modelo
# ============================================================
# accuracy indica el porcentaje de aciertos sobre el conjunto de prueba.
accuracy = accuracy_score(y_test, y_pred)

# La matriz de confusion muestra aciertos y errores por clase.
# Permite ver falsos positivos y falsos negativos, no solo el promedio.
conf_matrix = confusion_matrix(y_test, y_pred)

# El reporte incluye precision, recall y f1-score.
# Es util cuando las clases no estan balanceadas o importan tipos de error.
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# ============================================================
# 8. Visualizacion de probabilidades
# ============================================================
# Creamos 100 valores entre el minimo y maximo de horas para dibujar una curva.
study_hours_range = np.linspace(X.min(), X.max(), 100)

# predict_proba devuelve probabilidades para cada clase.
# [:, 1] toma la probabilidad de la clase 1, es decir, aprobar.
y_prob = model.predict_proba(study_hours_range.reshape(-1, 1))[:, 1]

# Puntos azules: observaciones reales del conjunto de prueba.
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Curva roja: probabilidad estimada de aprobar segun las horas de estudio.
plt.plot(study_hours_range, y_prob, color='red', label='Logistic Regression Curve')

# Etiquetas para interpretar el eje X, el eje Y y el significado del grafico.
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Study Hours vs. Pass/Fail')
plt.legend()

# Muestra la ventana del grafico.
plt.show()
