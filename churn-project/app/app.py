from flask import Flask, request, jsonify
import joblib
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# --- Configuración de rutas ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
DB_PATH    = os.path.join(os.path.dirname(__file__), 'predictions.db')

# --- Cargar el modelo al arranque ---
model = joblib.load(MODEL_PATH)

# FEATURES esperadas en el mismo orden que fue entrenado el modelo
FEATURE_NAMES = ['edad', 'uso_mensual', 'tickets_soporte', 'meses_contrato', 'pagos_atrasados']


def init_db():
    """Inicializa la base de datos SQLite con la tabla de historial."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            edad             INTEGER,
            uso_mensual      INTEGER,
            tickets_soporte  INTEGER,
            meses_contrato   INTEGER,
            pagos_atrasados  INTEGER,
            prediction       INTEGER,
            probabilidad     REAL
        )
    ''')
    conn.commit()
    conn.close()


# Inicializar la DB al importar el módulo (compatible con Gunicorn + Flask)
init_db()


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health-check para Docker y orquestadores (p. ej. Azure Container Apps)."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe un JSON con los features del cliente y retorna la predicción de churn.

    Body esperado:
    {
        "edad": 35,
        "uso_mensual": 3,
        "tickets_soporte": 6,
        "meses_contrato": 12,
        "pagos_atrasados": 2
    }
    """
    data = request.get_json(force=True)

    # --- Validación de features ---
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({'error': f'Faltan features: {missing}'}), 400

    features = [data[f] for f in FEATURE_NAMES]

    # --- Inferencia ---
    prediction   = int(model.predict([features])[0])
    probabilidad = float(model.predict_proba([features])[0][1])

    # --- MLOps: Inference Logging → SQLite ---
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history
            (timestamp, edad, uso_mensual, tickets_soporte, meses_contrato, pagos_atrasados, prediction, probabilidad)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow().isoformat(),
        data['edad'], data['uso_mensual'], data['tickets_soporte'],
        data['meses_contrato'], data['pagos_atrasados'],
        prediction, probabilidad
    ))
    conn.commit()
    conn.close()

    return jsonify({
        'churn_prediction':          prediction,
        'probabilidad_de_abandono':  round(probabilidad, 4),
        'riesgo':                    'ALTO' if probabilidad >= 0.5 else 'BAJO',
        'status':                    'Logged to Database'
    })


@app.route('/history', methods=['GET'])
def get_history():
    """Retorna las últimas 20 predicciones registradas."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, edad, uso_mensual, tickets_soporte,
               meses_contrato, pagos_atrasados, prediction, probabilidad
        FROM history ORDER BY id DESC LIMIT 20
    ''')
    columns = [desc[0] for desc in cursor.description]
    rows    = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return jsonify(rows)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)