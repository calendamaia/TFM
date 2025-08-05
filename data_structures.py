"""
Módulo que define las estructuras de datos fundamentales para el problema de timetabling.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class TimeTableSolution:
    """
    Clase que representa una solución al problema de asignación de tribunales.
    
    Attributes:
        chromosome (np.ndarray): Matriz de estudiantes x horarios donde cada celda
                               contiene 3 miembros del tribunal
        fitness (float): Valor de aptitud de la solución
    """
    chromosome: np.ndarray
    fitness: float = 0.0

    def __post_init__(self):
        """
        Validación posterior a la inicialización.
        """
        if not isinstance(self.chromosome, np.ndarray):
            raise ValueError("El cromosoma debe ser un array de numpy")

    def __eq__(self, other):
        """
        Implementa la comparación de igualdad entre soluciones.
        """
        if not isinstance(other, TimeTableSolution):
            return NotImplemented
        return (np.array_equal(self.chromosome, other.chromosome) and 
                self.fitness == other.fitness)

    def __hash__(self):
        """
        Implementa el hash de la solución para poder usarla en sets y diccionarios.
        """
        return hash((self.fitness, self.chromosome.tobytes()))

    def copy(self):
        """
        Crea una copia profunda de la solución.
        
        Returns:
            TimeTableSolution: Nueva instancia con los mismos valores
        """
        return TimeTableSolution(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness
        )