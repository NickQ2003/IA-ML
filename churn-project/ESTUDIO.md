# 📚 Plan de Estudio a Profundidad — Churn Prediction System

> **Objetivo**: Entender cada capa del sistema, desde los datos hasta el contenedor en producción.  
> **Tiempo estimado**: 4–6 horas en 5 módulos progresivos.

---

## 🗺️ Mapa del Sistema (Big Picture)

```
[ Datos Crudos ]
      ↓
[ train_model.py ]  →  genera  →  [ app/model.pkl ]
                                         ↓
                              [ app/app.py (Flask API) ]
                                    ↓         ↓
                          [ /predict ]   [ /history ]
                               ↓               ↓
                        [ predictions.db (SQLite) ]
                                    ↓
                          [ Dockerfile (Docker) ]
                                    ↓
                      [ Contenedor en localhost:5000 ]
                                    ↓
                         [ (Futuro) Azure / Cloud ]
```

---

## Módulo 1 — Los Datos y el Problema de Negocio

**Archivo clave**: `train_model.py`

### ¿Qué está pasando?

El script genera un **dataset sintético** de 300 clientes con 5 características (features). Estos datos simulan lo que en producción vendría de una base de datos real (CRM, ERP, etc.).

| Feature | ¿Qué representa? | ¿Por qué importa para churn? |
|---|---|---|
| `edad` | Edad del cliente | Clientes mayores pueden ser más conservadores |
| `uso_mensual` | Sesiones por mes | Poco uso = desenganche |  
| `tickets_soporte` | Tickets abiertos | Más quejas = más insatisfacción |
| `meses_contrato` | Antigüedad | Clientes nuevos se van más fácil |
| `pagos_atrasados` | Retrasos en pago | Señal de desinterés financiero |

### La lógica de Churn (léela despacio)

```python
score_churn = (
    (uso_mensual < 5).astype(int) * 2     +   # Peso 2: muy importante
    (tickets_soporte > 5).astype(int) * 2 +   # Peso 2: muy importante
    (pagos_atrasados > 2).astype(int) * 1 +   # Peso 1: moderado
    (meses_contrato < 6).astype(int) * 1      # Peso 1: moderado
)
churn = ((score_churn + ruido_aleatorio) >= 3).astype(int)
```

> **Concepto clave**: Estamos *asignando* la etiqueta `churn=1` basados en reglas de negocio. En producción, esta etiqueta vendría de datos históricos reales (clientes que SÍ se fueron).

### 🧪 Experimentos para hacer

> **Cómo ejecutar cualquier experimento:**
> 1. Abre `train_model.py` en VS Code
> 2. Aplica el cambio indicado
> 3. Desde la terminal en la carpeta del proyecto: `python train_model.py`
> 4. Lee el output y reflexiona sobre lo que cambió

---

#### Experimento 1 — Más datos = ¿mejor modelo?

**Qué cambiar** en `train_model.py`, línea 6:
```python
# ANTES
N = 300

# DESPUÉS
N = 1000
```

**Cómo ejecutar:**
```powershell
# En tu terminal, dentro de churn-project/
python train_model.py
```

**Qué observar en el output:**
```
Dataset generado: 1000 clientes | Tasa de churn: XX.X%

=== Reporte de Clasificación (Test Set) ===
              precision    recall  f1-score   support
           0       X.XX      X.XX      X.XX       ???   ← ¿Subió?
           1       X.XX      X.XX      X.XX       ???
    accuracy                           X.XX       200   ← Ahora son 200 casos de test
```

**Qué esperar:** La accuracy debería subir (o estabilizarse cerca del 85-90%). Con más datos, el modelo tiene más ejemplos para aprender patrones. El `support` de la clase 1 también aumenta, lo que hace las métricas más confiables.

**Reflexión:** ¿Por qué con 4 datos originales del proyecto base era imposible confiar en el modelo?

---

#### Experimento 2 — ¿Qué pasa si cambias los pesos del churn?

**Qué cambiar** en `train_model.py`, líneas 19-24:
```python
# ANTES (pagos_atrasados tiene peso 1)
score_churn = (
    (uso_mensual < 5).astype(int) * 2     +
    (tickets_soporte > 5).astype(int) * 2 +
    (pagos_atrasados > 2).astype(int) * 1 +
    (meses_contrato < 6).astype(int) * 1
)

# DESPUÉS (pagos_atrasados ahora tiene peso 3 — es el más importante)
score_churn = (
    (uso_mensual < 5).astype(int) * 1     +
    (tickets_soporte > 5).astype(int) * 1 +
    (pagos_atrasados > 2).astype(int) * 3 +
    (meses_contrato < 6).astype(int) * 1
)
```

**Cómo ejecutar:**
```powershell
python train_model.py
```

**Qué observar en el output — la sección de Feature Importance:**
```
=== Importancia de Features ===
  pagos_atrasados      X.XXX   ← ¿Subió al primer lugar?
  tickets_soporte      X.XXX
  uso_mensual          X.XXX
  ...
```

**Qué esperar:** El modelo aprende de los datos. Si `pagos_atrasados` ahora es el factor que más determina el `churn=1` en tus datos, el Random Forest lo detecta y le da mayor importancia automáticamente.

**Reflexión clave:** Esto demuestra que el modelo **no sabe de negocios** — aprende de los datos que le das. Si los datos están sesgados, el modelo también lo estará. En el mundo real, los pesos vienen de análisis histórico real, no de reglas inventadas.

---

#### Experimento 3 — Agregar una nueva feature

**Qué cambiar** en `train_model.py`. Hay **3 lugares** que tocar:

**Paso 1** — Genera la nueva columna (después de línea 15, `pagos_atrasados = ...`):
```python
# AGREGAR esta línea:
dias_sin_login = np.random.randint(0, 90, N)   # 0 a 90 días sin entrar
```

**Paso 2** — Inclúyela en el score de churn (modifica la fórmula):
```python
score_churn = (
    (uso_mensual < 5).astype(int) * 2      +
    (tickets_soporte > 5).astype(int) * 2  +
    (pagos_atrasados > 2).astype(int) * 1  +
    (meses_contrato < 6).astype(int) * 1   +
    (dias_sin_login > 60).astype(int) * 2  # ← NUEVA: más de 60 días = señal fuerte
)
```

**Paso 3** — Agrégala al DataFrame (modifica el `pd.DataFrame(...)`):
```python
df = pd.DataFrame({
    'edad': edad,
    'uso_mensual': uso_mensual,
    'tickets_soporte': tickets_soporte,
    'meses_contrato': meses_contrato,
    'pagos_atrasados': pagos_atrasados,
    'dias_sin_login': dias_sin_login,   # ← NUEVA columna
    'churn': churn
})
```

**Cómo ejecutar:**
```powershell
python train_model.py
```

**Qué observar:**
```
=== Importancia de Features ===
  dias_sin_login       X.XXX   ← ¿Aparece? ¿Con qué posición?
  tickets_soporte      X.XXX
  uso_mensual          X.XXX
  ...
```

**⚠️ Importante — también debes actualizar `app/app.py`:**  
El modelo ahora espera 6 features, pero la API solo envía 5. Debes agregar `dias_sin_login` a `FEATURE_NAMES`:
```python
# En app/app.py, línea ~13
FEATURE_NAMES = ['edad', 'uso_mensual', 'tickets_soporte',
                 'meses_contrato', 'pagos_atrasados', 'dias_sin_login']
```
Y actualizar el `INSERT INTO history` para incluir la nueva columna.  
**Esto simula exactamente lo que es un "schema migration" en producción.**

### ❓ Preguntas de comprensión

- ¿Por qué usamos `stratify=y` en el `train_test_split`?
- ¿Qué significa una tasa de churn del 37.3%? ¿Es mucho o poco?
- ¿Por qué guardamos el modelo con `joblib.dump` y no con `pickle`?

---

## Módulo 2 — El Modelo de Machine Learning (Random Forest)

**Archivo clave**: `train_model.py` (las últimas 15 líneas)

### ¿Qué es un Random Forest?

Es un **ensemble de árboles de decisión**. Cada árbol vota y la mayoría gana.

```
Cliente: [edad=45, uso=2, tickets=7, meses=5, pagos_tard=3]
         ↓         ↓         ↓         ↓ ... (100 árboles)
      Árbol 1   Árbol 2   Árbol 3   Árbol N
      churn=1   churn=1   churn=0   churn=1
          ↓
    Voto mayoritario → churn = 1 (ALTO RIESGO)
    predict_proba    → 97.08%
```

### Los hiperparámetros que usamos

```python
RandomForestClassifier(
    n_estimators=100,  # 100 árboles en el bosque
    max_depth=5,       # Cada árbol puede tener máximo 5 niveles
    random_state=42    # Semilla fija = resultados reproducibles
)
```

> **¿Por qué `max_depth=5`?** Sin límite, los árboles "memorizan" los datos de entrenamiento (overfitting). Con `max_depth=5` forzamos que el modelo **generalice**.

### Leyendo el Classification Report

```
              precision    recall  f1-score   support
           0       0.89      0.82      0.85        38   ← Clientes que NO se van
           1       0.72      0.82      0.77        22   ← Clientes que SÍ se van
    accuracy                           0.82        60
```

| Métrica | Fórmula | En negocio significa |
|---|---|---|
| **Precision** | TP / (TP + FP) | "De los que predije que se van, ¿cuántos realmente se fueron?" |
| **Recall** | TP / (TP + FN) | "De los que SÍ se fueron, ¿cuántos detecté?" |
| **F1-Score** | Promedio armónico | Balance entre precision y recall |

> **⚠️ Insight de negocio**: Para churn, el **Recall** es más importante que Precision. Es peor no detectar a un cliente que se va (FN) que contactar a uno que no se iba (FP).

### Feature Importance — Lo que el modelo aprendió

```
tickets_soporte   38.5%  ← El modelo aprendió que es el mejor predictor
uso_mensual       34.4%  ← Segundo más importante
edad              10.7%
meses_contrato     9.0%
pagos_atrasados    7.4%
```

### 🧪 Experimentos para hacer

1. Prueba `n_estimators=10` vs `n_estimators=500`. Mide el tiempo y la accuracy.
2. Prueba `max_depth=None` (sin límite). ¿Sube la accuracy? ¿Es esto bueno?
3. Cambia el clasificador de `RandomForestClassifier` a `LogisticRegression`. ¿Qué cambia?

### ❓ Preguntas de comprensión

- ¿Qué es `predict_proba` y en qué se diferencia de `predict`?
- Si `recall` de la clase 1 es 0.82, ¿cuántos de cada 100 clientes fugados detectamos?
- ¿Por qué dividimos en train/test y no usamos todos los datos para entrenar?

---

## Módulo 3 — La API REST con Flask

**Archivo clave**: `app/app.py`

### Anatomía de la API

```
app.py
  │
  ├── init_db()          → Crea la tabla en SQLite si no existe
  ├── model = joblib.load()  → Carga el modelo en memoria RAM al arrancar
  │
  ├── GET  /health       → ¿Está viva la API? ¿Cargó el modelo?
  ├── POST /predict      → Recibe features, devuelve predicción + la guarda en DB
  └── GET  /history      → Devuelve las últimas 20 predicciones
```

### El flujo de `/predict` paso a paso

```
1. POST { "edad": 45, "uso_mensual": 2, ... }
         ↓
2. request.get_json()  → Parsea el JSON del body
         ↓
3. Validación: ¿faltan features? → Error 400
         ↓
4. features = [45, 2, 7, 5, 3]  (lista ordenada)
         ↓
5. model.predict([features])       → [1]   (¿Se va? Sí)
   model.predict_proba([features]) → [0.029, 0.971]  (3% no, 97% sí)
         ↓
6. INSERT INTO history ...  → Guarda en SQLite
         ↓
7. return jsonify({...})    → Responde al cliente
```

### ¿Por qué `init_db()` está fuera del `if __name__`?

```python
# ❌ Solo funciona con `python app.py` directamente:
if __name__ == '__main__':
    init_db()       # ← Gunicorn NUNCA ejecuta este bloque
    app.run(...)

# ✅ Se ejecuta cuando cualquier proceso importa el módulo (incluido Gunicorn):
init_db()           # ← Esto sí funciona con Gunicorn

if __name__ == '__main__':
    app.run(...)    # Solo para desarrollo local
```

### ¿Qué es SQLite y por qué lo usamos?

SQLite es una **base de datos de archivo único** (`.db`). Es perfecta para:
- ✅ Prototipado y desarrollo
- ✅ Guardar historial de predicciones (Inference Logging)
- ❌ Múltiples usuarios simultáneos en producción (usar PostgreSQL en ese caso)

### 🧪 Experimentos para hacer

1. Llama a `/predict` sin el campo `tickets_soporte`. ¿Qué responde la API?
2. Llama a `/history` después de hacer 5 predicciones y observa el JSON.
3. Agrega un endpoint `GET /stats` que devuelva: total de predicciones, % de churn detectado.

### ❓ Preguntas de comprensión

- ¿Qué es `jsonify` y por qué no podemos devolver un dict directamente?
- ¿Por qué `FEATURE_NAMES` es una lista ordenada y no un set?
- ¿Qué pasa si el modelo no encuentra `model.pkl` al arrancar?

---

## Módulo 4 — MLOps: Inference Logging

**Concepto**: Guardar cada predicción en la DB es una práctica de **MLOps** llamada *Inference Logging*.

### ¿Para qué sirve el historial?

```
predictions.db → history table
  ↓
1. AUDITORÍA:   "¿Por qué el modelo dijo que este cliente se iba a ir?"
2. MONITOREO:   ¿La distribución de predicciones cambió vs. el mes pasado? (Model Drift)
3. RE-ENTRENAMIENTO: Los datos reales de clientes que SÍ se fueron → nuevo dataset
4. REPORTING:   El equipo de marketing puede ver quién está en riesgo
```

### El ciclo virtuoso de MLOps

```
Datos Históricos → Entrenamiento → Modelo → Predicciones en Producción
        ↑                                              ↓
        └─────── Nuevos datos etiquetados ←── predictions.db
```

> **Clave Senior**: Con el tiempo, el `predictions.db` se convierte en tu **dataset de re-entrenamiento**. Cuando tengas suficientes casos donde puedes verificar si el cliente realmente se fue, lo usas para mejorar el modelo.

### 🧪 Experimentos para hacer

1. Abre `predictions.db` con [DB Browser for SQLite](https://sqlitebrowser.org/) (es gratis). Visualiza la tabla `history`.
2. Escribe una query SQL directamente: `SELECT AVG(probabilidad) FROM history WHERE prediction=1`
3. Piensa: ¿cómo agregarías una columna `resultado_real` para registrar si el cliente SÍ se fue?

---

## Módulo 5 — Docker y Contenerización

**Archivo clave**: `Dockerfile`

### ¿Qué problema resuelve Docker?

> "En mi PC funciona" → "En Azure también funciona" ✅

Docker empaqueta **el código + sus dependencias + el sistema operativo base** en una imagen hermética.

### El Multi-Stage Build (nuestra arquitectura)

```dockerfile
# ── STAGE 1: TRAINER ──────────────────────────────
FROM python:3.11-slim AS trainer    # Imagen base de Python
WORKDIR /build
COPY requirements.txt .
RUN pip install ...                 # Instala sklearn, pandas, etc.
COPY train_model.py .
RUN python train_model.py           # ← ENTRENA EL MODELO AQUÍ (dentro de Docker)
# Resultado: /build/app/model.pkl

# ── STAGE 2: PRODUCTION ───────────────────────────
FROM python:3.11-slim AS production # Nueva imagen limpia (sin artefactos de entrenamiento)
WORKDIR /app
COPY requirements.txt .
RUN pip install ...
COPY app/ .                         # Copia el código de Flask
COPY --from=trainer /build/app/model.pkl ./model.pkl  # ← Trae el modelo del Stage 1
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", ...]
```

### ¿Por qué Multi-Stage?

| Sin Multi-Stage | Con Multi-Stage |
|---|---|
| La imagen incluye pandas, numpy, scikit-learn de entrenamiento | La imagen de producción es más liviana |
| El `model.pkl` debe existir ANTES del build | El modelo se genera DURANTE el build |
| Si alguien clona el repo sin el `.pkl`, la imagen crashea | Siempre funciona, self-contained |

### ¿Por qué Gunicorn en vez de `python app.py`?

| Flask dev server | Gunicorn |
|---|---|
| 1 solo proceso | 4 workers (procesos paralelos) |
| No apto para producción | Diseñado para producción |
| 1 request a la vez | 4 requests simultáneos |

### Comandos Docker esenciales

```powershell
# Construir la imagen
docker build -t churn-api:latest .

# Correr el contenedor
docker run -d -p 5000:5000 --name churn churn-api:latest

# Ver logs en tiempo real
docker logs -f churn

# Ver contenedores corriendo
docker ps

# Entrar al contenedor (para debug)
docker exec -it churn bash

# Detener y eliminar
docker rm -f churn
```

### 🧪 Experimentos para hacer

1. Corre `docker exec -it churn-test bash` (cuando el contenedor esté activo) y navega a `/app`. ¿Qué archivos ves?
2. Elimina el `model.pkl` de la carpeta `app/` y haz `docker build` de nuevo. ¿Funciona igual? ¿Por qué?
3. Cambia `--workers 4` a `--workers 1`. ¿Cambia algo en uso normal?

### ❓ Preguntas de comprensión

- ¿Qué es una imagen Docker vs. un contenedor Docker?
- ¿Por qué `EXPOSE 5000` no es suficiente para acceder al puerto? (necesitas `-p 5000:5000`)
- ¿Qué pasaría con `predictions.db` si eliminas y recreás el contenedor?

---

## 🎓 Evaluación Final — Preguntas Integradoras

Responde estas preguntas para validar que entiendes el sistema completo:

1. **Flujo end-to-end**: Si un agente de marketing quiere saber si un cliente se va a ir, ¿qué secuencia exacta de eventos ocurre desde que hace el request hasta que recibe la respuesta?

2. **Fallo de producción**: El `model.pkl` dentro del contenedor se corrompió. ¿Cuál es el proceso para actualizar el modelo sin perder el historial de predicciones?

3. **Escalabilidad**: El sistema ahora debe manejar 10,000 requests por minuto. ¿Qué dos cambios harías? (Pista: base de datos y workers)

4. **Re-entrenamiento**: Han pasado 6 meses. Tienes 5,000 predicciones en `predictions.db` y ya sabes cuáles clientes realmente se fueron. ¿Cómo reinventas el ciclo de entrenamiento?

5. **Azure**: Si quisieras desplegar `churn-api:latest` en Azure Container Apps, ¿cuáles serían los primeros 3 pasos?

---

## 📖 Recursos para Profundizar

| Tema | Recurso |
|---|---|
| Random Forest | [Scikit-learn Docs — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| Flask API | [Flask Quickstart](https://flask.palletsprojects.com/en/stable/quickstart/) |
| SQLite con Python | [sqlite3 Module Docs](https://docs.python.org/3/library/sqlite3.html) |
| Docker Multi-Stage | [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/) |
| Gunicorn | [Gunicorn Docs](https://gunicorn.org/) |
| MLOps Concepts | [Google MLOps Whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) |

---

*Generado en: 2026-03-12 | Proyecto: Churn Prediction System*
