# Modelos de Machine Learning (Churn Prediction)

Este repositorio contiene scripts básicos para entrenar y exportar modelos de predicción de abandono de clientes (Customer Churn) utilizando diferentes frameworks populares de Machine Learning y Deep Learning:

1. **PyTorch** (`pytorch/model.py`)
2. **Scikit-learn** (`Scikit-learn/model.py`)
3. **TensorFlow/Keras** (`TensorFlow/model.py`)

## Requisitos Previos

Asegúrate de tener Python instalado en tu sistema (versión 3.8 o superior recomendada).

### 1. Entorno Virtual (Recomendado)

Se recomienda utilizar un entorno virtual para instalar las dependencias de manera aislada:

```bash
# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual (Windows)
venv\Scripts\activate

# Activar el entorno virtual (Linux/Mac)
source venv/bin/activate
```

### 2. Instalación de Dependencias

Instala los paquetes necesarios definidos en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Preparar el Dataset

Todos los modelos asumen que existe un archivo llamado `customer_churn.csv` en la ubicación desde la que se ejecutan los scripts. 
En el repositorio actualmente hay un archivo CSV con un nombre largo (ej. `ZXZr6Kf...Dataset.csv`). **Debes renombrarlo a `customer_churn.csv`** para que los scripts funcionen sin modificaciones. 

### 4. Ejecución de los modelos

Para probar cualquiera de los modelos, navega al directorio del framework correspondiente y ejecuta el archivo `model.py` correspondiente. Es muy importante que `customer_churn.csv` se encuentre al alcance (puedes copiarlo dentro de la carpeta antes de ejecutar el script).

**Para PyTorch:**
```bash
cd pytorch
# Asegúrate de tener customer_churn.csv aquí
python model.py
```
*(Al finalizar el entrenamiento, se guardarán los pesos del modelo en el archivo `churn_model.pth`)*

**Para Scikit-learn (Random Forest):**
```bash
cd Scikit-learn
# Asegúrate de tener customer_churn.csv aquí
python model.py
```
*(Al finalizar, se guardará el modelo en formato binario mediante joblib en el archivo `churn_model.pkl`)*

**Para TensorFlow:**
```bash
cd TensorFlow
# Asegúrate de tener customer_churn.csv aquí
python model.py
```
*(El script entrenará una red neuronal secuencial, evaluará su accuracy y la exportará con el nombre `churn_model.h5`. También incluye el código base de conversión a TFLite)*

## Notas Técnicas
- El código de TensorFlow asume que dispones de `import tensorflow as tf` al principio del archivo (esto ha sido corregido).
- Dependiendo de tu hardware, es posible que TensorFlow y PyTorch instalen por defecto versiones de CPU. Si deseas utilizar aceleración GPU, tendrás que instalar las versiones específicas de cada framework que incluyan soporte de CUDA.
