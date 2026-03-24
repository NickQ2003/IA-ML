# 📚 Guía de Limpieza y Preprocesamiento de Datos
> **Nivel:** Estudiante | **Script:** `clear_data.py` | **Dataset:** `dataset_clientes.csv`

---

## ¿Qué es la limpieza de datos y por qué importa?

Antes de entrenar cualquier modelo de Machine Learning, los datos deben estar **limpios y en formato numérico**. En el mundo real, los datos suelen llegar con:

- Filas con información faltante (vacíos, `NaN`)
- Valores imposibles (edad = 999, salario = -5000)
- Texto inconsistente ("M", "Masculino", "m")
- Filas duplicadas

Un modelo entrenado con datos sucios aprende patrones incorrectos → **basura entra, basura sale**.

---

## 📦 PASO 0 — Importar librerías

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno
```

| Librería | ¿Para qué sirve? |
|---|---|
| `pandas` | Cargar, explorar y manipular tablas de datos |
| `numpy` | Operaciones matemáticas sobre arreglos |
| `sklearn` | Herramientas de Machine Learning (escaladores, encoders) |
| `missingno` | Visualizar gráficamente dónde faltan datos |

---

## 📂 PASO 1 — Cargar el dataset

```python
df = pd.read_csv(r'ruta\al\dataset_clientes.csv')
print(df.head())
```

### 🔍 Variable creada: `df`
Es un **DataFrame** (tabla). Piénsalo como una hoja de Excel en Python.

### 🛑 Pausa para explorar — ejecuta esto en consola:

```python
print(df.shape)          # (filas, columnas)
print(df.dtypes)         # tipo de dato de cada columna
print(df.info())         # resumen general
print(df.describe())     # estadísticas: min, max, media, etc.
print(df.isnull().sum()) # cuántos valores faltan por columna
print(df.duplicated().sum()) # cuántas filas duplicadas hay
```

> 💡 **Tip:** `df.head(10)` muestra las primeras 10 filas. `df.tail(5)` muestra las últimas 5.

---

## 🔎 PASO 2 — Visualizar datos faltantes

```python
msno.matrix(df)   # mapa visual de NaN (blanco = faltante)
msno.heatmap(df)  # correlación entre columnas con datos faltantes
```

Estas funciones abren una **gráfica** que te muestra de un vistazo qué columnas tienen huecos y si los datos faltantes aparecen juntos.

---

## 🗑️ PASO 3 — Manejar valores faltantes (NaN)

Hay dos estrategias principales:

### Estrategia A — Eliminar filas con NaN

```python
df_cleaned = df.dropna()
```

### 🔍 Variable creada: `df_cleaned`
Nueva tabla **sin filas incompletas**. El `df` original **no se modifica**.

```python
# 🛑 Pausa para comparar:
print("Filas originales:", df.shape[0])
print("Filas sin NaN:   ", df_cleaned.shape[0])
print("Filas eliminadas:", df.shape[0] - df_cleaned.shape[0])
```

> ⚠️ **Cuándo usarlo:** Cuando tienes muchos datos y pocas filas con NaN.
> Si eliminas el 40% de tu dataset, esta estrategia es mala.

---

### Estrategia B — Rellenar NaN (imputación)

```python
df_filled = df.copy()

# Columnas numéricas → rellenar con la MEDIA
df_filled[df.select_dtypes(include=[np.number]).columns] = (
    df_filled.select_dtypes(include=[np.number])
    .fillna(df.mean(numeric_only=True))
)

# Columnas de texto → rellenar con la MODA (valor más frecuente)
df_filled[df.select_dtypes(include=['object', 'category']).columns] = (
    df_filled.select_dtypes(include=['object', 'category'])
    .fillna(df.select_dtypes(include=['object', 'category']).mode().iloc[0])
)
```

### 🔍 Variables creadas:
| Variable | Qué contiene |
|---|---|
| `df_filled` | Copia del DataFrame con NaN reemplazados |

#### ¿Por qué dos bloques separados?
- `df.mean()` solo puede calcularse en **números**. Si incluyes texto, da `TypeError`.
- `.mode().iloc[0]` toma la primera fila del resultado de moda (puede haber empates).

```python
# 🛑 Pausa para verificar:
print(df_filled.isnull().sum())   # debe ser todo 0
print(df.mean(numeric_only=True)) # qué valores se usaron para rellenar
```

---

## 🏷️ PASO 4 — Identificar tipos de columnas

```python
numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
```

### 🔍 Variables creadas:
| Variable | Contenido esperado |
|---|---|
| `numerical_cols` | `['id', 'edad', 'salario', 'calificacion']` |
| `categorical_cols` | `['nombre', 'genero', 'ciudad', 'fecha_registro', 'email', 'categoria']` |

```python
# 🛑 Pausa para ver:
print("Numéricas:   ", numerical_cols)
print("Categóricas: ", categorical_cols)
```

> 💡 Los modelos de ML **solo entienden números**. Por eso necesitamos transformar las columnas de texto.

---

## ⚖️ PASO 5 — Estandarización y Normalización

Ambas técnicas cambian la **escala** de los números sin cambiar su distribución.

### A) Estandarización (StandardScaler)

```python
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

**Fórmula:** `z = (x - media) / desviación_estándar`

Resultado: valores con **media = 0** y **desviación estándar = 1**.

### B) Normalización (MinMaxScaler)

```python
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

**Fórmula:** `x' = (x - min) / (max - min)`

Resultado: todos los valores entre **0 y 1**.

```python
# 🛑 Pausa para ver el efecto:
print(df[numerical_cols].describe())
# Después de MinMaxScaler: min=0.0, max=1.0 en todas las columnas
```

> ⚠️ **Nota:** En este script se aplican ambos uno después del otro. En la práctica,
> elige **uno solo** según tu algoritmo:
> - **StandardScaler** → SVM, regresión logística, PCA
> - **MinMaxScaler** → redes neuronales, KNN

---

## 🔤 PASO 6 — Codificación de variables categóricas

Los algoritmos no entienden "Ciudad de México". Hay que **convertir texto a números**.

### Clasificación previa:

```python
# Se hace ANTES de get_dummies (muy importante)
large_categorical_cols = [col for col in categorical_cols if df[col].nunique() > 20]
small_categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 20]
```

| Variable | Qué guarda | Ejemplo |
|---|---|---|
| `large_categorical_cols` | Columnas con **más de 20 valores únicos** | `nombre`, `email` |
| `small_categorical_cols` | Columnas con **≤ 20 valores únicos** | `genero`, `ciudad`, `categoria` |

```python
# 🛑 Pausa para ver:
print("Columnas grandes (One-Hot):", large_categorical_cols)
print("Columnas pequeñas (Label):  ", small_categorical_cols)
```

---

### A) One-Hot Encoding (columnas grandes)

```python
df = pd.get_dummies(df, columns=large_categorical_cols, drop_first=True)
```

Crea **una columna binaria (0/1) por cada valor único**.

**Antes:**

| email |
|---|
| ana@email.com |
| carlos@email.com |

**Después:**

| email_carlos@email.com | email_maria@email.com | ... |
|---|---|---|
| 0 | 0 | ... |
| 1 | 0 | ... |

> ⚠️ `drop_first=True` elimina la primera columna para evitar multicolinealidad.

---

### B) Label Encoding (columnas pequeñas)

```python
from sklearn.preprocessing import LabelEncoder
for col in small_categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
```

Asigna un **número entero a cada categoría**.

**Antes:**

| genero |
|---|
| F |
| M |
| F |

**Después:**

| genero |
|---|
| 0 |
| 1 |
| 0 |

```python
# 🛑 Pausa para ver la transformación de una columna:
# (ejecuta ANTES del LabelEncoder para ver el antes)
print(df['genero'].value_counts())
# (ejecuta DESPUÉS para ver el después)
print(df['genero'].unique())
```

> ⚠️ Label Encoding solo funciona bien con columnas **ordinales** (A < B < C)
> o binarias (F/M). Para categorías sin orden, One-Hot es mejor.

---

## 🚨 PASO 7 — Detección y corrección de Outliers

```python
outlier_threshold = 3  # umbral en desviaciones estándar

for col in numerical_cols:
    mean = df[col].mean()
    std  = df[col].std()
    df[col] = np.where(
        np.abs(df[col] - mean) > outlier_threshold * std,
        mean,     # Si es outlier → reemplaza con la media
        df[col]   # Si no → déjalo igual
    )
```

### ¿Cómo funciona la regla de las 3σ?

Visualmente:

```
       ←── 99.7% de los datos ───→
  ──|──────────|──────────|──────────|──────────|──────────|──
 -3σ          -2σ        -1σ        μ         +1σ        +2σ        +3σ
Todo lo que quede FUERA de [-3σ, +3σ] se considera OUTLIER
```

**Ejemplo con nuestro dataset:**
- `edad = 999` → Outlier (la media es ~37, el 999 está a cientos de σ)
- `salario = -5000` → Outlier (salario negativo imposible)

```python
# 🛑 Pausa para detectar outliers antes de corregirlos:
for col in numerical_cols:
    mean = df[col].mean()
    std  = df[col].std()
    outliers = df[np.abs(df[col] - mean) > 3 * std]
    if len(outliers) > 0:
        print(f"\nOutliers en '{col}':")
        print(outliers[[col]])
```

---

## 💾 PASO 8 — Guardar el dataset limpio

```python
df.to_csv('cleaned_dataset.csv', index=False)
```

Guarda el DataFrame procesado como `cleaned_dataset.csv` en el mismo directorio.
`index=False` evita que pandas escriba el índice numérico del DataFrame como columna extra.

---

## 🗺️ Resumen Visual del Pipeline

```
dataset_clientes.csv
        │
        ▼
┌───────────────────┐
│  1. Cargar datos  │  df = pd.read_csv(...)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────┐
│  2. Explorar y visualizar │  df.info(), msno.matrix()
└────────┬──────────────────┘
         │
         ▼
┌────────────────────────────────┐
│  3. Manejar valores faltantes  │  dropna() / fillna()
└────────┬───────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  4. Escalar columnas numéricas       │  StandardScaler, MinMaxScaler
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  5. Codificar columnas categóricas   │  One-Hot, Label Encoding
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  6. Tratar outliers              │  Regla 3σ
└────────┬─────────────────────────┘
         │
         ▼
  cleaned_dataset.csv  ✅
```

---

## 🧪 Hoja de trucos — Comandos para inspeccionar en cualquier momento

```python
# Ver dimensiones
df.shape                        # (filas, columnas)

# Ver tipos de dato
df.dtypes

# Ver primeras/últimas filas
df.head(5)
df.tail(5)

# Ver valores únicos de una columna
df['genero'].unique()
df['genero'].value_counts()

# Ver estadísticas
df.describe()                   # solo numéricas
df.describe(include='all')      # todas las columnas

# Ver valores faltantes
df.isnull().sum()
df.isnull().mean() * 100        # en porcentaje

# Ver duplicados
df.duplicated().sum()
df[df.duplicated()]             # ver las filas duplicadas

# Ver outliers en una columna
col = 'edad'
mean, std = df[col].mean(), df[col].std()
df[np.abs(df[col] - mean) > 3 * std]
```

---

*Generado para el curso de Limpieza y Preprocesamiento de Datos — IA & ML*
