# ============================================================
# Clase practica: eliminacion hacia atras
# ============================================================
# Objetivo de esta tecnica:
# Empezar con todas las variables disponibles y eliminar poco a poco las que
# parecen aportar menos al modelo segun su p-value.
#
# Idea central:
# La seleccion hacia atras parte de un modelo completo. En cada iteracion mira
# que variable tiene menor evidencia estadistica y la elimina si supera un
# umbral de significancia.
#
# Esta tecnica responde la pregunta:
# "Si empiezo usando todas las variables, cuales puedo quitar porque no aportan
# suficiente informacion al modelo?"

# ============================================================
# 1. Importacion de librerias
# ============================================================
# pandas organiza datos tabulares.
# statsmodels permite ajustar modelos estadisticos y ver p-values.
# train_test_split esta importado, aunque en este ejemplo no se usa.
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# ============================================================
# 2. Dataset de ejemplo
# ============================================================
# Cada fila representa un estudiante.
# Las columnas StudyHours y PrevExamScore son variables explicativas.
# Pass es la variable objetivo.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convertimos los datos a DataFrame para trabajar con columnas.
df = pd.DataFrame(data)

# ============================================================
# 3. Separacion entre entradas y salida
# ============================================================
# X contiene todas las variables candidatas al inicio.
X = df[['StudyHours', 'PrevExamScore']]

# y contiene la variable que queremos explicar.
y = df['Pass']

# ============================================================
# 4. Preparacion del modelo estadistico
# ============================================================
# statsmodels no agrega automaticamente el intercepto.
# sm.add_constant agrega una columna constante para que el modelo tenga termino
# independiente, equivalente al intercepto en una regresion.
X = sm.add_constant(X)

# OLS significa Ordinary Least Squares, o minimos cuadrados ordinarios.
# Ajusta una regresion lineal y permite inspeccionar coeficientes y p-values.
model = sm.OLS(y, X).fit()

# El summary muestra una tabla estadistica completa del modelo inicial.
# Aqui lo importante para esta tecnica son los p-values de cada variable.
print(model.summary())

# ============================================================
# 5. Nivel de significancia
# ============================================================
# significance_level define el umbral para decidir si una variable se queda.
# Con 0.05, una variable con p-value mayor a 0.05 se considera poco significativa
# bajo esta regla practica.
significance_level = 0.05

# ============================================================
# 6. Eliminacion hacia atras
# ============================================================
# Repetimos el proceso hasta que todas las variables restantes tengan p-value
# menor o igual al nivel de significancia.
while True:
    # Ajustamos el modelo con las variables que siguen disponibles.
    model = sm.OLS(y, X).fit()

    # Buscamos el p-value mas alto del modelo actual.
    # Ese valor indica la variable con menor evidencia estadistica relativa.
    max_p_value = model.pvalues.max()
    
    # Si el p-value mas alto supera el umbral, eliminamos esa variable.
    if max_p_value > significance_level:
        # Identificamos el nombre de la variable con peor p-value.
        feature_to_remove = model.pvalues.idxmax()
        print(f"Removing feature: {feature_to_remove} with p-value: {max_p_value}")
        
        # Eliminamos la columna del conjunto de variables.
        X = X.drop(columns=[feature_to_remove])
    else:
        # Si ninguna variable supera el umbral, terminamos.
        break

# ============================================================
# 7. Modelo final
# ============================================================
# Mostramos el resumen del modelo despues de eliminar variables.
print(model.summary())

# ============================================================
# 8. Como reflexionar sobre esta tecnica
# ============================================================
# - Ventaja: ayuda a simplificar un modelo eliminando variables debiles.
# - Riesgo: los p-values dependen de supuestos estadisticos, tamano de muestra
#   y relaciones entre variables.
# - Cuidado: con datasets muy pequenos, como este, las conclusiones no deben
#   interpretarse como definitivas.
# - Pregunta practica: estoy eliminando variables por falta real de informacion
#   o porque tengo pocos datos para medir bien su efecto?
