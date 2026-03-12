import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# --- Semilla para reproducibilidad ---
np.random.seed(42)
N = 300  # Número de clientes simulados

# --- Ingeniería de Características (Feature Engineering) ---
# Simulamos un dataset más realista con 5 features relevantes para churn
edad            = np.random.randint(22, 65, N)
uso_mensual     = np.random.randint(0, 30, N)      # Sesiones por mes
tickets_soporte = np.random.randint(0, 10, N)      # Tickets de soporte abiertos
meses_contrato  = np.random.randint(1, 48, N)      # Antigüedad del cliente
pagos_atrasados = np.random.randint(0, 5, N)       # Veces que pagó tarde

# --- Lógica de Churn Realista ---
# Un cliente tiende a irse si: usa poco el servicio, abre muchos tickets y paga tarde
score_churn = (
    (uso_mensual < 5).astype(int) * 2     +
    (tickets_soporte > 5).astype(int) * 2 +
    (pagos_atrasados > 2).astype(int) * 1 +
    (meses_contrato < 6).astype(int) * 1
)
# Churn = 1 si el score es >= 3 (con algo de ruido aleatorio)
churn = ((score_churn + np.random.randint(0, 2, N)) >= 3).astype(int)

df = pd.DataFrame({
    'edad': edad,
    'uso_mensual': uso_mensual,
    'tickets_soporte': tickets_soporte,
    'meses_contrato': meses_contrato,
    'pagos_atrasados': pagos_atrasados,
    'churn': churn
})

print(f"Dataset generado: {N} clientes | Tasa de churn: {churn.mean():.1%}")

# --- Entrenamiento del Modelo ---
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluación ---
print("\n=== Reporte de Clasificación (Test Set) ===")
print(classification_report(y_test, model.predict(X_test)))

# Importancia de features (MLOps: entender qué mueve la aguja)
print("=== Importancia de Features ===")
for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.3f}")

# --- Persistencia del Modelo ---
os.makedirs('app', exist_ok=True)
joblib.dump(model, 'app/model.pkl')
print("\nModelo guardado en app/model.pkl")