# ============================================================
# Comparacion de modelos de clasificacion
# ============================================================
# Objetivo del archivo:
# Entrenar dos modelos sobre el mismo problema y comparar su rendimiento.
# Esto ayuda a entender que no existe un unico algoritmo "mejor"; se evalua
# segun datos, metricas, interpretabilidad y comportamiento esperado.

# ============================================================
# 1. Importacion de librerias
# ============================================================
# LogisticRegression representa un modelo lineal probabilistico.
# DecisionTreeClassifier representa un modelo basado en reglas.
# Las metricas permiten comparar ambos con los mismos criterios.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# ============================================================
# 2. Creacion del dataset
# ============================================================
# Cada fila representa un estudiante con dos variables explicativas.
# La etiqueta Pass indica si aprobo o reprobo.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

# Convertimos los datos a una tabla para separar columnas facilmente.
df = pd.DataFrame(data)

# Inspeccion rapida de estructura y valores.
print(df.head())

# ============================================================
# 3. Separacion entre variables de entrada y salida
# ============================================================
# Ambos modelos recibiran las mismas caracteristicas para que la comparacion
# sea justa: horas de estudio y puntaje previo.
X = df[['StudyHours', 'PrevExamScore']]

# La variable objetivo es la clase real que queremos predecir.
y = df['Pass']

# ============================================================
# 4. Division en entrenamiento y prueba
# ============================================================
# Usar la misma particion para ambos modelos permite comparar resultados
# bajo las mismas condiciones.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")

# ============================================================
# 5. Modelo 1: Regresion logistica
# ============================================================
# Este modelo aprende una frontera de decision relativamente simple.
# Es util cuando la separacion entre clases puede aproximarse de forma lineal.
logreg_model = LogisticRegression()

# Entrenamos el modelo con los datos de entrenamiento.
logreg_model.fit(X_train, y_train)

# Predecimos sobre los datos de prueba para evaluar generalizacion.
y_pred_logreg = logreg_model.predict(X_test)

# Calculamos accuracy como primera metrica comparativa.
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")

# ============================================================
# 6. Modelo 2: Arbol de decision
# ============================================================
# El arbol aprende reglas explicitas sobre las variables.
# Suele ser facil de interpretar, pero puede sobreajustar si crece demasiado.
tree_model = DecisionTreeClassifier(random_state=42)

# Entrenamos el arbol con la misma particion de entrenamiento.
tree_model.fit(X_train, y_train)

# Predecimos las clases para los mismos datos de prueba.
y_pred_tree = tree_model.predict(X_test)

# Calculamos accuracy para comparar contra la regresion logistica.
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree}")

# ============================================================
# 7. Evaluacion detallada de la regresion logistica
# ============================================================
# Miramos no solo cuantos aciertos hubo, sino tambien el tipo de errores.
print("Logistic Regression:")
print(f"Accuracy: {accuracy_logreg}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print("Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# ============================================================
# 8. Evaluacion detallada del arbol de decision
# ============================================================
# Usamos las mismas metricas para que la comparacion sea consistente.
print("Decision Tree:")
print(f"Accuracy: {accuracy_tree}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))

# ============================================================
# 9. Visualizacion del arbol
# ============================================================
# Solo el arbol se grafica porque su estructura de reglas es interpretable.
# La regresion logistica tambien se puede visualizar, pero requiere graficar
# probabilidades o frontera de decision.
plt.figure(figsize=(12,8))
tree.plot_tree(tree_model, feature_names=['StudyHours', 'PrevExamScore'], class_names=['Fail', 'Pass'], filled=True)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.show()

