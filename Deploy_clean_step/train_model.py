from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Cargar dataset Iris
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42
)

# Entrenar modelo
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, "iris_model.pkl")

print("Modelo entrenado y guardado como iris_model.pkl")