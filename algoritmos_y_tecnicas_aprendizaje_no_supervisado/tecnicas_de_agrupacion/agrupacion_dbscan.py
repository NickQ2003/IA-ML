"""Ejemplo didactico de agrupacion usando DBSCAN.

DBSCAN es un algoritmo de aprendizaje no supervisado basado en densidad.
No necesita que indiquemos el numero de clusters desde el inicio. En cambio,
usa la distancia entre puntos para descubrir zonas densas y marca como ruido
los puntos que no pertenecen claramente a ningun grupo.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class AgrupacionDBSCAN:
    """Agrupa clientes con DBSCAN usando ingreso anual y puntaje de gasto."""

    def __init__(self, eps=0.5, min_samples=3):
        # eps es la distancia maxima para considerar que dos puntos son vecinos.
        # min_samples indica cuantos puntos cercanos se necesitan para formar
        # una zona densa. Cambiar estos valores cambia el resultado del modelo.
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.modelo = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.datos = None
        self.datos_escalados = None

    def crear_datos(self):
        """Crea el conjunto de datos de clientes para el ejemplo."""
        # AnnualIncome representa el ingreso anual en miles.
        # SpendingScore representa un puntaje de gasto de 1 a 100.
        # Los ultimos tres registros son valores atipicos para ver como DBSCAN
        # puede separarlos o marcarlos como ruido.
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
        """Normaliza las columnas para que las distancias sean comparables."""
        # DBSCAN calcula distancias entre puntos. Si una columna tiene una escala
        # mayor que otra, puede dominar el calculo. StandardScaler evita eso.
        datos_escalados = self.scaler.fit_transform(self.datos)
        self.datos_escalados = pd.DataFrame(
            datos_escalados,
            columns=self.datos.columns,
        )
        return self.datos_escalados

    def entrenar_modelo(self):
        """Entrena DBSCAN y agrega la etiqueta de cluster al DataFrame."""
        # fit_predict ejecuta el algoritmo y devuelve una etiqueta por fila.
        # La etiqueta -1 significa ruido: puntos que no entran en un cluster.
        etiquetas = self.modelo.fit_predict(self.datos_escalados)
        self.datos["Cluster"] = etiquetas
        return self.datos

    def contar_clusters(self):
        """Cuenta los clusters encontrados sin incluir el ruido."""
        etiquetas = set(self.datos["Cluster"])
        return len(etiquetas - {-1})

    def graficar_clusters(self, titulo):
        """Muestra los clientes coloreados segun el cluster asignado."""
        plt.scatter(
            self.datos["AnnualIncome"],
            self.datos["SpendingScore"],
            c=self.datos["Cluster"],
            cmap="rainbow",
        )
        plt.title(titulo)
        plt.xlabel("Ingreso anual en miles")
        plt.ylabel("Puntaje de gasto")
        plt.show()

    def cambiar_parametros(self, eps, min_samples=None):
        """Permite probar DBSCAN con otros parametros."""
        # Este metodo facilita comparar resultados sin repetir todo el codigo.
        self.eps = eps
        if min_samples is not None:
            self.min_samples = min_samples

        self.modelo = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def ejecutar(self):
        """Ejecuta el flujo completo del ejemplo con dos valores de eps."""
        self.crear_datos()
        print("Datos originales:")
        print(self.datos.head())

        self.escalar_datos()
        print("\nDatos escalados:")
        print(self.datos_escalados.head())

        self.entrenar_modelo()
        print("\nResultado con eps=0.5:")
        print(self.datos.head())
        print(f"Clusters encontrados: {self.contar_clusters()}")
        self.graficar_clusters("DBSCAN con eps=0.5")

        self.cambiar_parametros(eps=0.7)
        self.entrenar_modelo()
        print("\nResultado con eps=0.7:")
        print(self.datos.head())
        print(f"Clusters encontrados: {self.contar_clusters()}")
        self.graficar_clusters("DBSCAN con eps=0.7")


if __name__ == "__main__":
    ejemplo = AgrupacionDBSCAN(eps=0.5, min_samples=3)
    ejemplo.ejecutar()
