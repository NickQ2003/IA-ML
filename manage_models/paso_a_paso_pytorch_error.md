# Solución al Error de PyTorch

Al intentar ejecutar el script `pytorch/model.py`, te has encontrado con el siguiente error en la terminal:

```python
TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

## ¿Por qué sucedió este error?

1. **Preprocesamiento de Pandas:** 
   El dataset original `customer_churn.csv` contiene columnas categóricas como `Contract` y `PaymentMethod`. Al usar `pd.get_dummies(data, drop_first=True)` para transformar estas columnas en valores numéricos, Pandas versión 3.x convierte estas nuevas variables en formato Booleano (`True` o `False`).
   
2. **Mezcla de tipos de datos en la matriz de Numpy:**
   Cuando intentas extraer los valores para dárselos a PyTorch (usando `X_train.values`), Numpy nota que tienes una mezcla de números decimales (`float64`), enteros (`int64`) y booleanos (`bool`). Para acomodarlos todos en un solo arreglo sin perder información, Numpy crea un arreglo genérico de tipo `object` (`numpy.object_`).

3. **Restricción de PyTorch:**
   A diferencia de Scikit-learn o Keras que hacen conversiones automáticas por debajo, la función `torch.tensor()` de PyTorch es estricta con los tipos de datos y **no soporta el tipo genérico `object`**. Exige un arreglo con datos homogéneos y puramente numéricos. Por esto, el código colapsaba justo en la línea:
   ```python
   outputs = model(torch.tensor(X_train.values).float())
   ```

## ¿Cómo se solucionó?

El paso a paso para la solución aplicada fue forzar a Pandas/Numpy a convertir todo el arreglo en números decimales estándar ANTES de pasárselo a PyTorch.

**Código Antiguo:**
```python
outputs = model(torch.tensor(X_train.values).float())
loss = criterion(outputs.squeeze(), torch.tensor(y_train.values).float())

# ... y para test ...
outputs = model(torch.tensor(X_test.values).float())
```

**Código Corregido:**
```python
outputs = model(torch.tensor(X_train.astype(float).values).float())
loss = criterion(outputs.squeeze(), torch.tensor(y_train.astype(float).values).float())

# ... y para test ...
outputs = model(torch.tensor(X_test.astype(float).values).float())
```

Al agregar `.astype(float)`, forzamos a que los valores `True` pasen a ser `1.0`, los `False` a `0.0`, y evitamos por completo que se genere el arreglo de tipo `object`.

> **Nota:** Ya he aplicado este arreglo directamente a tu archivo `pytorch/model.py`. Si vuelves a correr el modelo (estando en tu entorno virtual), ahora funcionará perfectamente.
