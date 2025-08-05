"""
Módulo principal para la asignación optimizada de tribunales y horarios de TFG/TFM

Este módulo implementa la interfaz de usuario y el flujo principal del programa para resolver
el problema de asignación de tribunales y horarios utilizando algoritmos metaheurísticos.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
from genetic_algorithm import TimetablingGA
from harmony_search import TimetablingHS
from visualization import (
    generate_professional_plots,
    generate_additional_plots
)

def run_algorithm(algorithm_class, input_file: str):
    """
    Ejecuta un algoritmo de optimización específico y retorna sus resultados.
    
    Args:
        algorithm_class: Clase del algoritmo a ejecutar (TimetablingGA o TimetablingHS)
        input_file (str): Ruta al archivo Excel con los datos de entrada
    
    Returns:
        tuple: Tupla con (mejor_solución, historial_fitness, tiempo_total,
               número_generaciones, tiempo_inicio)
    """
    import time
    start_time = time.time()
    algorithm = algorithm_class(input_file)
    best_solution, fitness_history = algorithm.solve()
    end_time = time.time()
    total_time = end_time - start_time
    generations = len(fitness_history)
    
    return best_solution, fitness_history, total_time, generations, start_time

def main():
    """
    Función principal del programa que gestiona la ejecución completa del proceso
    de optimización de horarios.
    """
    try:
        # Mostrar encabezado
        print("="*80)
        print("SISTEMA DE OPTIMIZACIÓN PARA LA ASIGNACIÓN DE TRIBUNALES Y HORARIOS DE TFG/TFM")
        print("="*80)
        print("\nUniversidad Isabel I")
        print("-"*80)
        
        # Solicitar archivo de entrada
        input_file = input("\nPor favor, ingrese el nombre del archivo Excel de entrada "
                          "(o presione Enter para usar el valor por defecto): ").strip()
        
        if not input_file:
            input_file = "DatosGestionTribunales.xlsx"
            print(f"\nUsando el archivo por defecto: {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"\nError: El archivo '{input_file}' no existe en el directorio actual.")
        
        # Verificar logo
        logo_path = 'logoui1.png'
        if not os.path.exists(logo_path):
            print(f"\nAdvertencia: No se encuentra el archivo del logo ({logo_path})")
            print("Las gráficas se generarán sin marca de agua.")
        
        # Crear directorio para resultados
        results_dir = "resultados_comparacion"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Generar timestamp único para los archivos
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ejecutar algoritmos
        print("\nEjecutando Algoritmo Genético...")
        print("-"*50)
        try:
            ga_results = run_algorithm(TimetablingGA, input_file)
        except Exception as e:
            raise RuntimeError(f"Error en la ejecución del Algoritmo Genético: {str(e)}")
        
        print("\nEjecutando Harmony Search...")
        print("-"*50)
        try:
            hs_results = run_algorithm(TimetablingHS, input_file)
        except Exception as e:
            raise RuntimeError(f"Error en la ejecución del Harmony Search: {str(e)}")
        
        # Generar nombres de archivos
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        ga_output = os.path.join(results_dir, f"solucionAG-{base_filename}-{timestamp}.xlsx")
        hs_output = os.path.join(results_dir, f"solucionHS-{base_filename}-{timestamp}.xlsx")
        
        # Exportar soluciones
        print("\nExportando soluciones...")
        print("-"*50)
        try:
            ga_algorithm = TimetablingGA(input_file)
            ga_algorithm.export_solution(ga_results[0], ga_output)
            
            hs_algorithm = TimetablingHS(input_file)
            hs_algorithm.export_solution(hs_results[0], hs_output)
        except Exception as e:
            raise RuntimeError(f"Error al exportar soluciones: {str(e)}")
        
        # Generar visualizaciones
        print("\nGenerando análisis comparativo con gráficas profesionales...")
        print("-"*50)
        try:
            generate_professional_plots(
                ga_results, hs_results, results_dir,
                f"{base_filename}_{timestamp}", logo_path
            )
            generate_additional_plots(
                ga_results, hs_results, results_dir,
                f"{base_filename}_{timestamp}", logo_path
            )
            
            print("\nProceso completado exitosamente.")
            print(f"\nLos resultados se han guardado en el directorio: {results_dir}")
            
        except Exception as e:
            print(f"\nError al generar gráficas: {str(e)}")
            print("\nSe han guardado las soluciones aunque hubo errores en la generación de gráficas.")
            print(f"\nLos resultados se han guardado en el directorio: {results_dir}")
    
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
    
    finally:
        plt.close('all')
        print("\nPrograma finalizado.")

if __name__ == "__main__":
    main()