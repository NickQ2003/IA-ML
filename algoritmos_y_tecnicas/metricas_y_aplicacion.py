# ============================================================
# Clase practica: metricas y aplicacion en modelos de ML
# ============================================================
# Objetivo de esta clase:
# Entender que entrenar un modelo no es suficiente. Tambien necesitamos
# medir su rendimiento con metricas adecuadas para saber si realmente esta
# aprendiendo patrones utiles o si solo parece funcionar en un caso puntual.
#
# En este archivo se trabajan dos tipos de problemas:
# 1. Clasificacion: predecir una clase, por ejemplo aprobar (1) o reprobar (0).
# 2. Regresion: predecir un valor numerico continuo, por ejemplo una nota.
#
# Idea clave para reflexionar:
# Una metrica no es solo un numero. Es una forma de hacer una pregunta sobre
# el comportamiento del modelo: cuantos aciertos tiene, que tipo de errores
# comete y que tan estable es cuando cambia la muestra de datos.

# ============================================================
# 1. Importacion de librerias
# ============================================================
# numpy se usa para calculos numericos, como promediar resultados.
# pandas permite organizar los datos en forma de tabla.
# scikit-learn aporta modelos, division de datos, validacion cruzada y metricas.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score

# ============================================================
# 2. Dataset de ejemplo para clasificacion
# ============================================================
# Cada fila representa un estudiante.
# StudyHours: horas de estudio.
# PrevExamScore: puntaje previo del estudiante.
# Pass: resultado real que queremos predecir.
#
# En clasificacion binaria normalmente usamos:
# 0 para la clase negativa, en este caso reprobar.
# 1 para la clase positiva, en este caso aprobar.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convertimos el diccionario a DataFrame para trabajar con columnas.
df = pd.DataFrame(data)

# ============================================================
# 3. Separacion entre entradas X y salida y
# ============================================================
# X contiene las variables que el modelo observa para aprender.
# Aqui el modelo usara horas de estudio y puntaje previo.
X = df[['StudyHours', 'PrevExamScore']]

# y contiene la respuesta correcta que el modelo debe predecir.
y = df['Pass']

from sklearn.linear_model import LogisticRegression

# ============================================================
# 4. Division en entrenamiento y prueba
# ============================================================
# Entrenamiento: datos que el modelo usa para aprender.
# Prueba: datos separados para evaluar si el modelo generaliza.
#
# Pregunta para reflexionar:
# Si evaluamos con los mismos datos usados para entrenar, estamos midiendo
# aprendizaje real o solo memorizacion?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 5. Entrenamiento de un modelo de clasificacion
# ============================================================
# LogisticRegression aprende una relacion entre las variables de entrada
# y la probabilidad de pertenecer a la clase positiva, aqui aprobar.
model = LogisticRegression()

# fit significa ajustar o entrenar: el modelo aprende desde X_train/y_train.
model.fit(X_train, y_train)

# predict genera clases finales sobre datos que el modelo no uso para entrenar.
y_pred = model.predict(X_test)

# ============================================================
# 6. Metricas de clasificacion
# ============================================================
# Accuracy responde: de todas las predicciones, que proporcion fue correcta?
# Es facil de entender, pero puede ser enganosa si una clase domina el dataset.
accuracy = accuracy_score(y_test, y_pred)

# Precision responde: cuando el modelo predice clase positiva, cuantas veces
# tiene razon? Sirve cuando los falsos positivos son costosos.
# Ejemplo: decir que alguien aprueba cuando realmente reprueba.
precision = precision_score(y_test, y_pred)

# Recall responde: de todos los casos positivos reales, cuantos encontro?
# Sirve cuando los falsos negativos son costosos.
# Ejemplo: no detectar a un estudiante que realmente si iba a aprobar.
recall = recall_score(y_test, y_pred)

# F1 combina precision y recall en una sola medida.
# Es util cuando queremos balancear ambos tipos de error.
f1 = f1_score(y_test, y_pred)

# Imprimimos las metricas para comparar resultados numericamente.
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

from sklearn.model_selection import cross_val_score

# ============================================================
# 7. Validacion cruzada con una metrica
# ============================================================
# Problema de una sola division train/test:
# el resultado puede depender mucho de que filas cayeron en entrenamiento
# y cuales cayeron en prueba.
#
# La validacion cruzada divide el dataset varias veces. En cada vuelta entrena
# con una parte y evalua con otra. Asi obtenemos una medida mas estable.
model = LogisticRegression()

# cv=5 significa 5 particiones o folds.
# scoring='accuracy' indica que queremos evaluar exactitud en cada fold.
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# cv_scores contiene una accuracy por cada fold.
# El promedio resume el rendimiento general bajo varias divisiones.
print(f'Cross-validation accuracies: {cv_scores}')
print(f'Mean cross-validation accuracy: {np.mean(cv_scores)}')


from sklearn.model_selection import cross_validate

# ============================================================
# 8. Validacion cruzada con varias metricas
# ============================================================
# cross_validate permite calcular varias metricas en el mismo proceso.
# Esto es mejor que mirar solo accuracy, porque distintos errores pueden tener
# distinta importancia segun el problema real.
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Ejecutamos validacion cruzada usando las metricas definidas arriba.
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

# Cada clave test_* guarda los resultados obtenidos en los folds.
# Usamos np.mean para obtener una lectura promedio de cada metrica.
print(f"Cross-validation Accuracy: {np.mean(cv_results['test_accuracy'])}")
print(f"Cross-validation Precision: {np.mean(cv_results['test_precision'])}")
print(f"Cross-validation Recall: {np.mean(cv_results['test_recall'])}")
print(f"Cross-validation F1-Score: {np.mean(cv_results['test_f1'])}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 9. Ejemplo de regresion y metrica R2
# ============================================================
# Ahora cambiamos la pregunta:
# En vez de predecir si aprueba o reprueba, intentamos predecir un valor
# numerico: el puntaje previo usando las horas de estudio.
#
# Esto ya no es clasificacion, es regresion.
X_reg = df[['StudyHours']]
y_reg = df['PrevExamScore']

# LinearRegression aprende una linea para aproximar la relacion entre horas
# de estudio y puntaje previo.
reg_model = LinearRegression()

# R2 responde: que proporcion de la variacion del objetivo explica el modelo?
# Un R2 cercano a 1 suele indicar buen ajuste; cercano a 0 indica que explica
# poco; negativo puede indicar peor rendimiento que predecir el promedio.
cv_scores_r2 = cross_val_score(reg_model, X_reg, y_reg, cv=5, scoring='r2')

# Mostramos el R2 de cada fold y su promedio.
# Reflexion importante: con datasets pequenos, las metricas pueden variar mucho
# y no deben interpretarse como conclusiones definitivas.
print(f'Cross-validation R-squared scores: {cv_scores_r2}')
print(f'Mean R-squared score: {np.mean(cv_scores_r2)}')

# ============================================================
# 10. Como estudiar este archivo
# ============================================================
# 1. Primero identifica el tipo de problema: clasificacion o regresion.
# 2. Luego identifica que representa X y que representa y.
# 3. Despues pregunta: que modelo se entrena y con que datos?
# 4. Finalmente interpreta las metricas como preguntas sobre los errores.
#
# Resumen conceptual:
# - Accuracy: cuantos aciertos totales hubo.
# - Precision: que tan confiables son las predicciones positivas.
# - Recall: cuantos positivos reales logra encontrar el modelo.
# - F1: balance entre precision y recall.
# - Cross-validation: estabilidad del rendimiento en varias particiones.
# - R2: capacidad explicativa de un modelo de regresion.

