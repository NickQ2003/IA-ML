# ============================================================
# Clase practica: seleccion hacia adelante
# ============================================================
# Objetivo de esta tecnica:
# Elegir variables predictoras poco a poco, empezando con ninguna variable
# y agregando en cada paso la que mas mejora el rendimiento del modelo.
#
# Idea central:
# No siempre conviene usar todas las variables disponibles. Algunas variables
# pueden no aportar informacion, pueden meter ruido o pueden hacer que el
# modelo sea mas dificil de interpretar.
#
# Esta tecnica responde la pregunta:
# "Si empiezo desde cero, que variable agrego primero, despues cual agrego,
# y cuando dejo de agregar porque ya no mejora el modelo?"

# ============================================================
# 1. Importacion de librerias
# ============================================================
# pandas organiza los datos en forma de tabla.
# LinearRegression es el modelo que se usa para probar combinaciones.
# train_test_split separa entrenamiento y prueba.
# r2_score mide que tan bien explica el modelo la variable objetivo.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# 2. Dataset de ejemplo
# ============================================================
# Cada fila representa un estudiante.
# StudyHours y PrevExamScore son posibles variables predictoras.
# Pass es la variable objetivo que queremos explicar o predecir.
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

# Convertimos el diccionario a DataFrame para trabajar por columnas.
df = pd.DataFrame(data)

# ============================================================
# 3. Separacion entre variables de entrada y salida
# ============================================================
# X contiene las variables candidatas que la tecnica puede seleccionar.
X = df[['StudyHours', 'PrevExamScore']]

# y es la respuesta real que queremos predecir.
y = df['Pass']

# ============================================================
# 4. Funcion de seleccion hacia adelante
# ============================================================
# La funcion recibe:
# - X: tabla con todas las variables candidatas.
# - y: variable objetivo.
#
# Devuelve:
# - selected_features: lista con las variables que fueron elegidas.
def forward_selection(X, y):
    # remaining_features guarda las variables que todavia no han sido probadas
    # como seleccion final. Usamos set para poder remover variables facilmente.
    remaining_features = set(X.columns)

    # selected_features empieza vacia porque la tecnica arranca sin variables.
    selected_features = []

    # current_score guarda el mejor rendimiento conseguido hasta el momento.
    current_score = 0.0
    best_score = 0.0
    
    # Mientras existan variables candidatas, intentamos agregar una mas.
    while remaining_features:
        scores_with_candidates = []
        
        # Probamos cada variable restante como posible siguiente seleccion.
        for feature in remaining_features:
            # Combinamos las variables ya elegidas con una candidata nueva.
            features_to_test = selected_features + [feature]

            # Entrenamos y evaluamos usando solo esa combinacion temporal.
            X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)
            
            # Creamos y entrenamos un modelo lineal para medir esa combinacion.
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predecimos sobre prueba y calculamos R2.
            # R2 mas alto significa mejor capacidad explicativa en este ejemplo.
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            # Guardamos el resultado junto con la variable probada.
            scores_with_candidates.append((score, feature))
        
        # Ordenamos de mejor a peor rendimiento.
        scores_with_candidates.sort(reverse=True)

        # Tomamos la mejor variable candidata de esta ronda.
        best_score, best_feature = scores_with_candidates[0]
        
        # Solo agregamos la variable si mejora el score actual.
        # Si no mejora, detenemos el proceso porque seguir agregando no aporta.
        if current_score < best_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_score
        else:
            break
    
    return selected_features

# ============================================================
# 5. Ejecucion de la tecnica
# ============================================================
# Ejecutamos la funcion y mostramos que variables fueron seleccionadas.
best_features = forward_selection(X, y)
print("Selected features using Forward Selection:", best_features)

# ============================================================
# 6. Como reflexionar sobre esta tecnica
# ============================================================
# - Ventaja: construye un modelo simple agregando solo variables utiles.
# - Riesgo: puede quedarse con una decision local y no encontrar la mejor
#   combinacion global de variables.
# - Pregunta practica: cada variable agregada mejora realmente el modelo o solo
#   mejora por casualidad en una particion pequena de datos?

