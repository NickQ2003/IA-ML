# ============================================================
# Aplicacion de arboles de decision
# ============================================================
# Objetivo del archivo:
# Aprender un modelo supervisado de clasificacion que toma decisiones
# mediante reglas del tipo: "si una variable cumple cierta condicion,
# entonces avanzar por una rama del arbol".

# ============================================================
# 1. Importacion de librerias
# ============================================================
# pandas organiza los datos, scikit-learn aporta el arbol, la division
# train/test y las metricas, matplotlib permite visualizar la estructura.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# ============================================================
# 2. Creacion del dataset
# ============================================================
# Cada fila representa un estudiante.
# El modelo usara horas de estudio y nota previa para clasificar si aprueba.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Convertimos el diccionario a tabla para trabajar por columnas.
df = pd.DataFrame(data)

# Inspeccion inicial de los datos.
print(df.head())

# ============================================================
# 3. Separacion entre variables de entrada y salida
# ============================================================
# X contiene dos caracteristicas: horas estudiadas y nota anterior.
X = df[['StudyHours', 'PrevExamScore']]

# y contiene la etiqueta real: 0 = reprueba, 1 = aprueba.
y = df['Pass']

# ============================================================
# 4. Division en entrenamiento y prueba
# ============================================================
# El arbol aprende reglas con entrenamiento y se valida con datos separados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificamos tamanos de entrada y salida en ambos conjuntos.
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# ============================================================
# 5. Creacion y entrenamiento del arbol
# ============================================================
# DecisionTreeClassifier aprende cortes sobre las variables para separar clases.
# random_state hace reproducible el resultado cuando hay empates o aleatoriedad.
model = DecisionTreeClassifier(random_state=42)

# fit construye el arbol usando los datos de entrenamiento.
model.fit(X_train, y_train)

# La profundidad indica cuantos niveles de decisiones tiene el arbol.
# Las hojas son decisiones finales: puntos donde el arbol ya clasifica.
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

# ============================================================
# 6. Prediccion
# ============================================================
# predict recorre el arbol para cada estudiante de prueba y devuelve 0 o 1.
y_pred = model.predict(X_test)

# Comparacion directa entre resultado estimado y resultado real.
print("Predicted Outcomes (Pass/Fail):", y_pred)
print("Actual Outcomes:", y_test.values)

# ============================================================
# 7. Evaluacion del modelo
# ============================================================
# accuracy resume el porcentaje de predicciones correctas.
accuracy = accuracy_score(y_test, y_pred)

# La matriz de confusion permite ver errores especificos por clase.
conf_matrix = confusion_matrix(y_test, y_pred)

# classification_report agrega precision, recall y f1-score.
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# ============================================================
# 8. Visualizacion del arbol
# ============================================================
# El grafico permite leer las reglas aprendidas por el modelo.
# filled=True colorea nodos segun la clase dominante.
plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.show()

# ============================================================
# 9. Ajuste simple de hiperparametros
# ============================================================
# max_depth limita la profundidad del arbol.
# Esto puede reducir sobreajuste cuando un arbol memoriza demasiado los datos.
model_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)

# Entrenamos el modelo ajustado con los mismos datos de entrenamiento.
model_tuned.fit(X_train, y_train)

# Predecimos con el arbol limitado para comparar rendimiento.
y_pred_tuned = model_tuned.predict(X_test)

# Calculamos accuracy del modelo ajustado.
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f"Accuracy (Tuned Model): {accuracy_tuned}")

