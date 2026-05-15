"""Ejemplo didactico de agrupacion usando K-Means.

El objetivo del archivo es mostrar, paso a paso, el flujo tipico de un
algoritmo de aprendizaje no supervisado:

1. Crear o cargar datos.
2. Escalar las variables para que tengan una escala comparable.
3. Entrenar el algoritmo de agrupacion.
4. Agregar la etiqueta de cluster al conjunto original.
5. Visualizar los resultados.
6. Usar el metodo del codo para elegir un numero razonable de clusters.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AgrupacionKMeans:
    """Agrupa clientes con K-Means usando ingreso anual y puntaje de gasto."""

    def __init__(self, numero_clusters=3):
        # K-Means necesita que indiquemos cuantas agrupaciones queremos crear.
        # En este ejemplo se usan 3 clusters como punto de partida.
        self.numero_clusters = numero_clusters
        self.scaler = StandardScaler()
        self.modelo = KMeans(
            n_clusters=self.numero_clusters,
            random_state=42,
            n_init=10,
        )
        self.datos = None
        self.datos_escalados = None

    def crear_datos(self):
        """Crea un conjunto de datos pequeno para practicar agrupacion."""
        # AnnualIncome representa el ingreso anual del cliente en miles.
        # SpendingScore representa un puntaje de gasto entre 1 y 100.
        # Los ultimos valores tienen comportamientos extremos para observar
        # como el algoritmo los asigna a los clusters.
        data = {
            "AnnualIncome": [
                15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
                20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
                25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
                30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
                35,
                80, 85, 90,
            ],
            "SpendingScore": [
                39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
                69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
                6, 9, 12, 15, 18, 21, 24, 27, 30, 33,
                5, 8, 11, 14, 17, 20, 23, 26, 29, 32,
                56,
                2, 3, 100,
            ],
        }

        self.datos = pd.DataFrame(data)
        return self.datos

    def escalar_datos(self):
        """Normaliza los datos para que las variables tengan peso comparable."""
        # Si una variable tiene numeros mucho mas grandes que otra, K-Means
        # puede darle demasiada importancia. StandardScaler transforma cada
        # columna para que tenga media cercana a 0 y desviacion estandar 1.
        datos_escalados = self.scaler.fit_transform(self.datos)
        self.datos_escalados = pd.DataFrame(
            datos_escalados,
            columns=self.datos.columns,
        )
        return self.datos_escalados

    def entrenar_modelo(self):
        """Entrena K-Means y asigna cada cliente a un cluster."""
        # fit_predict entrena el modelo y devuelve la etiqueta del cluster
        # asignado a cada fila del DataFrame escalado.
        etiquetas = self.modelo.fit_predict(self.datos_escalados)
        self.datos["Cluster"] = etiquetas
        return self.datos

    def graficar_clusters(self):
        """Muestra una grafica de dispersion coloreada por cluster."""
        plt.scatter(
            self.datos["AnnualIncome"],
            self.datos["SpendingScore"],
            c=self.datos["Cluster"],
            cmap="viridis",
        )
        plt.title("Agrupacion K-Means de clientes")
        plt.xlabel("Ingreso anual en miles")
        plt.ylabel("Puntaje de gasto")
        plt.show()

    def calcular_wcss(self, maximo_clusters=10):
        """Calcula el WCSS para varios valores de k."""
        # WCSS mide que tan compactos son los clusters. Normalmente baja cuando
        # aumenta k, pero buscamos el punto donde la mejora empieza a ser menor.
        valores_wcss = []

        for cantidad_clusters in range(1, maximo_clusters + 1):
            modelo = KMeans(
                n_clusters=cantidad_clusters,
                random_state=42,
                n_init=10,
            )
            modelo.fit(self.datos_escalados)
            valores_wcss.append(modelo.inertia_)

        return valores_wcss

    def graficar_metodo_codo(self):
        """Grafica el metodo del codo para apoyar la eleccion de k."""
        valores_wcss = self.calcular_wcss()

        plt.plot(range(1, 11), valores_wcss, marker="o")
        plt.title("Metodo del codo para elegir k")
        plt.xlabel("Numero de clusters")
        plt.ylabel("WCSS")
        plt.show()

    def ejecutar(self):
        """Ejecuta el flujo completo del ejemplo."""
        self.crear_datos()
        print("Datos originales:")
        print(self.datos.head())

        self.escalar_datos()
        print("\nDatos escalados:")
        print(self.datos_escalados.head())

        self.entrenar_modelo()
        print("\nDatos con cluster asignado:")
        print(self.datos.head())

        self.graficar_clusters()
        self.graficar_metodo_codo()


if __name__ == "__main__":
    ejemplo = AgrupacionKMeans(numero_clusters=3)
    ejemplo.ejecutar()
