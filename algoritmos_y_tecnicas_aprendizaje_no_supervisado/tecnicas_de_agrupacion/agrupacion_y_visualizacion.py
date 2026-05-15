"""Ejemplo didactico de agrupacion y visualizacion.

Este documento compara dos algoritmos de clustering sobre el mismo conjunto de
datos: K-Means y DBSCAN. El objetivo es entender el flujo completo: preparar
datos, escalar variables, entrenar modelos, agregar etiquetas y graficar.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


class AgrupacionYVisualizacion:
    """Compara K-Means y DBSCAN usando datos de clientes."""

    def __init__(self, numero_clusters=3, eps=0.5, min_samples=3):
        # K-Means usa numero_clusters porque requiere definir cuantas
        # agrupaciones buscar. DBSCAN usa eps y min_samples porque trabaja con
        # densidad de puntos.
        self.numero_clusters = numero_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.modelo_kmeans = KMeans(
            n_clusters=self.numero_clusters,
            random_state=42,
            n_init=10,
        )
        self.modelo_dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.datos = None
        self.datos_escalados = None

    def crear_datos(self):
        """Crea el DataFrame base con ingresos y puntajes de gasto."""
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
        """Escala las variables numericas antes de entrenar los modelos."""
        # Escalar es importante porque ambos algoritmos usan distancias. Si las
        # columnas tienen escalas diferentes, el resultado puede quedar sesgado.
        datos_escalados = self.scaler.fit_transform(self.datos)
        self.datos_escalados = pd.DataFrame(
            datos_escalados,
            columns=self.datos.columns,
        )
        return self.datos_escalados

    def entrenar_kmeans(self):
        """Entrena K-Means y guarda sus etiquetas en el DataFrame."""
        # K-Means intenta formar grupos compactos alrededor de centros llamados
        # centroides. Cada punto queda asignado al centroide mas cercano.
        self.datos["KMeans_Cluster"] = self.modelo_kmeans.fit_predict(
            self.datos_escalados
        )
        return self.datos

    def entrenar_dbscan(self):
        """Entrena DBSCAN y guarda sus etiquetas en el DataFrame."""
        # DBSCAN busca regiones densas. Los puntos con etiqueta -1 son ruido.
        self.datos["DBSCAN_Cluster"] = self.modelo_dbscan.fit_predict(
            self.datos_escalados
        )
        return self.datos

    def graficar_kmeans(self):
        """Grafica los clusters creados por K-Means."""
        plt.scatter(
            self.datos["AnnualIncome"],
            self.datos["SpendingScore"],
            c=self.datos["KMeans_Cluster"],
            cmap="viridis",
        )
        plt.title("K-Means: agrupacion de clientes")
        plt.xlabel("Ingreso anual en miles")
        plt.ylabel("Puntaje de gasto")
        plt.show()

    def graficar_dbscan(self):
        """Grafica los clusters creados por DBSCAN."""
        plt.scatter(
            self.datos["AnnualIncome"],
            self.datos["SpendingScore"],
            c=self.datos["DBSCAN_Cluster"],
            cmap="rainbow",
        )
        plt.title("DBSCAN: agrupacion de clientes")
        plt.xlabel("Ingreso anual en miles")
        plt.ylabel("Puntaje de gasto")
        plt.show()

    def ejecutar(self):
        """Ejecuta todo el flujo de comparacion y visualizacion."""
        self.crear_datos()
        print("Datos originales:")
        print(self.datos.head())

        self.escalar_datos()
        print("\nDatos escalados:")
        print(self.datos_escalados.head())

        self.entrenar_kmeans()
        print("\nDatos con cluster de K-Means:")
        print(self.datos.head())
        self.graficar_kmeans()

        self.entrenar_dbscan()
        print("\nDatos con cluster de K-Means y DBSCAN:")
        print(self.datos.head())
        self.graficar_dbscan()


if __name__ == "__main__":
    ejemplo = AgrupacionYVisualizacion(
        numero_clusters=3,
        eps=0.5,
        min_samples=3,
    )
    ejemplo.ejecutar()
