"""
Script para ejecutar los algoritmos de optimización sobre conjuntos de datos sintéticos.

Este módulo implementa un procesador por lotes que ejecuta los algoritmos de optimización
(Algoritmo Genético y Harmony Search) sobre conjuntos de datos generados sintéticamente.
El procesador incluye control de tiempo de ejecución y generación de resultados comparativos.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
Fecha: 2024
"""

# Imports de bibliotecas estándar
import os
import sys
import shutil
import re
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set

def find_project_root() -> Path:
    """
    Encuentra el directorio raíz del proyecto (tfm/TFM).
    
    La función busca un directorio llamado 'tfm' o 'TFM' (insensible a mayúsculas)
    navegando hacia arriba desde la ubicación actual del archivo.
    
    Returns:
        Path: Ruta al directorio raíz del proyecto
        
    Raises:
        FileNotFoundError: Si no se encuentra el directorio raíz del proyecto
    
    Autor: Juan José Jiménez González
    Universidad: Universidad Isabel I
    """
    current = Path(__file__).resolve().parent
    
    # Buscar directorio tfm/TFM (insensible a mayúsculas)
    while current != current.parent:
        if current.name.lower() == 'tfm':
            return current
        current = current.parent
    
    # Si no se encuentra, lanzar excepción
    raise FileNotFoundError(
        "No se pudo encontrar el directorio raíz del proyecto (tfm/TFM). "
        f"Directorio actual: {Path(__file__).resolve().parent}"
)

# Configuración inicial del proyecto
project_root = find_project_root()
sys.path.append(str(project_root))

# Imports del proyecto
from genetic_algorithm import TimetablingGA
from harmony_search import TimetablingHS
from visualization import generate_professional_plots, generate_additional_plots

class BatchProcessor:
    """Clase base para el procesamiento por lotes de optimización de horarios."""
    
    def __init__(self, data_dir: str):
        """
        Inicializa el procesador por lotes.
        
        Args:
            data_dir: Directorio que contiene los datos a procesar
        """
        self.data_dir = Path(data_dir)
        self.results_dir = self.data_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger = None
        self.setup_logging()
        self.logger.info(f"Iniciando procesamiento en: {self.data_dir}")
        self.logger.info(f"Resultados se guardarán en: {self.results_dir}")

    def setup_logging(self):
        """Configura el sistema de logging."""
        log_file = self.results_dir / "batch_processing.log"
        
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.INFO)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def extract_scenario_info(self, log_content: str, excel_file: Path) -> str:
        """Extrae la información del escenario del log original."""
        try:
            scenario_match = re.search(r'DatosGestionTribunales-(\d+)\.xlsx', excel_file.name)
            if not scenario_match:
                return "Información no disponible"
            
            scenario_num = scenario_match.group(1)
            pattern = f"Escenario {scenario_num}:.*?(?=Escenario|$)"
            match = re.search(pattern, log_content, re.DOTALL)
            
            if match:
                return match.group(0).strip()
            return "Información no disponible"
            
        except Exception as e:
            self.logger.error(f"Error extrayendo información del escenario: {str(e)}")
            return f"Error extrayendo información: {str(e)}"

    def process_algorithms(self, excel_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Ejecuta los algoritmos GA y HS para un archivo dado."""
        ga_metrics = {}
        hs_metrics = {}
        
        try:
            # Ejecutar GA
            self.logger.info(f"Iniciando ejecución de GA para {excel_file.name}")
            try:
                start_time = time.time()
                ga = TimetablingGA(excel_path=str(excel_file))
                # Añadir logging de diagnóstico
                self.logger.info(f"Dimensiones del problema GA:")
                self.logger.info(f"Estudiantes: {ga.num_students}")
                self.logger.info(f"Profesores: {ga.num_professors}")
                self.logger.info(f"Turnos: {ga.num_timeslots}")
                
                try:
                    ga_solution, ga_fitness_history = ga.solve()
                    ga_time = time.time() - start_time
                    
                    ga_metrics = {
                        'tiempo_ejecucion': ga_time,
                        'fitness_final': ga_solution.fitness,
                        'generaciones': len(ga_fitness_history),
                        'mejor_fitness': max(ga_fitness_history),
                        'fitness_promedio': sum(ga_fitness_history) / len(ga_fitness_history),
                        'fitness_history': ga_fitness_history,
                        'solution': ga_solution
                    }
                    self.logger.info(f"GA completado exitosamente para {excel_file.name}")
                except Exception as solve_error:
                    self.logger.error(f"Error en GA.solve(): {str(solve_error)}")
                    self.logger.error(f"Detalles del error:", exc_info=True)
                    return {}, {}
                    
            except Exception as e:
                self.logger.error(f"Error en GA: {str(e)}")
                self.logger.error(f"Detalles del error:", exc_info=True)
                return {}, {}
                
            # Ejecutar HS
            self.logger.info(f"Iniciando ejecución de HS para {excel_file.name}")
            try:
                start_time = time.time()
                hs = TimetablingHS(excel_path=str(excel_file))
                # Añadir logging de diagnóstico
                self.logger.info(f"Dimensiones del problema HS:")
                self.logger.info(f"Estudiantes: {hs.num_students}")
                self.logger.info(f"Profesores: {hs.num_professors}")
                self.logger.info(f"Turnos: {hs.num_timeslots}")
                
                try:
                    hs_solution, hs_fitness_history = hs.solve()
                    hs_time = time.time() - start_time
                    
                    hs_metrics = {
                        'tiempo_ejecucion': hs_time,
                        'fitness_final': hs_solution.fitness,
                        'generaciones': len(hs_fitness_history),
                        'mejor_fitness': max(hs_fitness_history),
                        'fitness_promedio': sum(hs_fitness_history) / len(hs_fitness_history),
                        'fitness_history': hs_fitness_history,
                        'solution': hs_solution
                    }
                    self.logger.info(f"HS completado exitosamente para {excel_file.name}")
                except Exception as solve_error:
                    self.logger.error(f"Error en HS.solve(): {str(solve_error)}")
                    self.logger.error(f"Detalles del error:", exc_info=True)
                    return ga_metrics, {}
                    
            except Exception as e:
                self.logger.error(f"Error en HS: {str(e)}")
                self.logger.error(f"Detalles del error:", exc_info=True)
                return ga_metrics, {}
            
        except Exception as e:
            self.logger.error(f"Error general ejecutando algoritmos: {str(e)}")
            self.logger.error(f"Detalles del error:", exc_info=True)
            return {}, {}
            
        return ga_metrics, hs_metrics

    def process_single_file(self, excel_file: Path, log_content: str) -> bool:
        """
        Procesa un único archivo Excel.
        
        Args:
            excel_file: Archivo Excel a procesar
            log_content: Contenido del archivo log.txt
            
        Returns:
            bool: True si el procesamiento fue exitoso, False en caso contrario
        """
        try:
            # Crear directorio para resultados
            result_subdir = self.results_dir / excel_file.stem
            result_subdir.mkdir(exist_ok=True)
            
            # Copiar archivo original
            shutil.copy2(excel_file, result_subdir)
            
            # Crear log específico
            log_file = result_subdir / "processing_log.txt"
            
            # Ejecutar algoritmos y obtener métricas
            self.logger.info(f"Ejecutando algoritmos para {excel_file.name}")
            
            # Ejecutar GA
            start_time = time.time()
            ga = TimetablingGA(str(excel_file))
            ga_solution, ga_fitness_history = ga.solve()
            ga_time = time.time() - start_time
            ga_generations = len(ga_fitness_history)
            ga_output = result_subdir / f"solucion_GA_{excel_file.name}"
            ga.export_solution(ga_solution, str(ga_output))
            self.logger.info(f"Solución GA exportada a: {ga_output}")
            
            # Ejecutar HS
            start_time = time.time()
            hs = TimetablingHS(str(excel_file))
            hs_solution, hs_fitness_history = hs.solve()
            hs_time = time.time() - start_time
            hs_generations = len(hs_fitness_history)
            hs_output = result_subdir / f"solucion_HS_{excel_file.name}"
            hs.export_solution(hs_solution, str(hs_output))
            self.logger.info(f"Solución HS exportada a: {hs_output}")
            
            # Generar visualizaciones
            from visualization import generate_professional_plots, generate_additional_plots
            
            ga_results = (ga_solution, ga_fitness_history, ga_time, ga_generations, start_time)
            hs_results = (hs_solution, hs_fitness_history, hs_time, hs_generations, start_time)
            
            # Intentar obtener el logo
            logo_path = None
            try:
                project_root = Path(__file__).resolve().parent
                possible_logo_paths = [
                    project_root / "logoui1.png",
                    project_root.parent / "logoui1.png"
                ]
                
                for path in possible_logo_paths:
                    if path.exists():
                        logo_path = str(path)
                        break
            except Exception as e:
                self.logger.warning(f"No se pudo encontrar el logo: {str(e)}")
            
            try:
                generate_professional_plots(
                    ga_results, 
                    hs_results, 
                    str(result_subdir),
                    excel_file.stem,
                    logo_path
                )
                
                generate_additional_plots(
                    ga_results,
                    hs_results,
                    str(result_subdir),
                    excel_file.stem,
                    logo_path
                )
                
                self.logger.info("Gráficas generadas exitosamente")
                
            except Exception as e:
                self.logger.error(f"Error generando gráficas: {str(e)}")
            
            # Escribir log de procesamiento
            with open(log_file, "w", encoding='utf-8') as f:
                f.write("=== Información del Escenario Original ===\n")
                scenario_info = self.extract_scenario_info(log_content, excel_file)
                f.write(scenario_info + "\n\n")
                
                f.write("=== Ejecución de Algoritmos ===\n")
                
                # Calcular valores normalizados
                num_students = len(pd.read_excel(excel_file, sheet_name='Disponibilidad-alumnos-turnos'))
                max_fitness = num_students * 4  # Valor máximo posible del fitness
                
                # Métricas GA
                ga_metrics = {
                    'tiempo_ejecucion': ga_time,
                    'generaciones': ga_generations,
                    'fitness_final': ga_solution.fitness,
                    'mejor_fitness': max(ga_fitness_history),
                    'fitness_promedio': sum(ga_fitness_history) / len(ga_fitness_history)
                }
                
                # Escribir métricas GA
                f.write("\nMétricas Algoritmo Genético:\n")
                f.write(f"Número de estudiantes: {num_students}\n")
                f.write("Fitness:\n")
                if 'fitness_final' in ga_metrics:
                    fitness_norm = ga_metrics['fitness_final'] / max_fitness
                    f.write(f"  - Valor absoluto: {ga_metrics['fitness_final']:.6f}\n")
                    f.write(f"  - Valor normalizado [0-1]: {fitness_norm:.6f}\n")
                
                if 'mejor_fitness' in ga_metrics:
                    mejor_fitness_norm = ga_metrics['mejor_fitness'] / max_fitness
                    f.write(f"  - Mejor valor absoluto: {ga_metrics['mejor_fitness']:.6f}\n")
                    f.write(f"  - Mejor valor normalizado [0-1]: {mejor_fitness_norm:.6f}\n")
                
                if 'fitness_promedio' in ga_metrics:
                    promedio_norm = ga_metrics['fitness_promedio'] / max_fitness
                    f.write(f"  - Promedio absoluto: {ga_metrics['fitness_promedio']:.6f}\n")
                    f.write(f"  - Promedio normalizado [0-1]: {promedio_norm:.6f}\n")
                
                f.write(f"Tiempo de ejecución: {ga_metrics.get('tiempo_ejecucion', 'N/A')} segundos\n")
                f.write(f"Generaciones: {ga_metrics.get('generaciones', 'N/A')}\n")
                
                # Métricas HS
                hs_metrics = {
                    'tiempo_ejecucion': hs_time,
                    'generaciones': hs_generations,
                    'fitness_final': hs_solution.fitness,
                    'mejor_fitness': max(hs_fitness_history),
                    'fitness_promedio': sum(hs_fitness_history) / len(hs_fitness_history)
                }
                
                # Escribir métricas HS
                f.write("\nMétricas Harmony Search:\n")
                f.write(f"Número de estudiantes: {num_students}\n")
                f.write("Fitness:\n")
                if 'fitness_final' in hs_metrics:
                    fitness_norm = hs_metrics['fitness_final'] / max_fitness
                    f.write(f"  - Valor absoluto: {hs_metrics['fitness_final']:.6f}\n")
                    f.write(f"  - Valor normalizado [0-1]: {fitness_norm:.6f}\n")
                
                if 'mejor_fitness' in hs_metrics:
                    mejor_fitness_norm = hs_metrics['mejor_fitness'] / max_fitness
                    f.write(f"  - Mejor valor absoluto: {hs_metrics['mejor_fitness']:.6f}\n")
                    f.write(f"  - Mejor valor normalizado [0-1]: {mejor_fitness_norm:.6f}\n")
                
                if 'fitness_promedio' in hs_metrics:
                    promedio_norm = hs_metrics['fitness_promedio'] / max_fitness
                    f.write(f"  - Promedio absoluto: {hs_metrics['fitness_promedio']:.6f}\n")
                    f.write(f"  - Promedio normalizado [0-1]: {promedio_norm:.6f}\n")
                
                f.write(f"Tiempo de ejecución: {hs_metrics.get('tiempo_ejecucion', 'N/A')} segundos\n")
                f.write(f"Generaciones: {hs_metrics.get('generaciones', 'N/A')}\n")
            
            self.logger.info(f"Procesamiento completado para {excel_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error procesando {excel_file.name}: {str(e)}")
            return False
    
    def process_all_files(self):
        """Procesa todos los archivos Excel en el directorio."""
        try:
            # Leer el log original
            log_file = self.data_dir / "log.txt"
            if not log_file.exists():
                raise FileNotFoundError(f"No se encuentra log.txt en {self.data_dir}")

            encodings = ['cp1252', 'utf-8', 'latin-1']
            log_content = None
            
            for encoding in encodings:
                try:
                    with open(log_file, "r", encoding=encoding) as f:
                        log_content = f.read()
                        self.logger.info(f"Archivo log.txt leído correctamente con codificación {encoding}")
                        break
                except UnicodeDecodeError:
                    continue

            if log_content is None:
                raise UnicodeDecodeError("No se pudo decodificar el archivo log.txt con ninguna codificación")

            # Procesar cada archivo Excel
            excel_files = list(self.data_dir.glob("*.xlsx"))
            total_files = len(excel_files)

            self.logger.info(f"Encontrados {total_files} archivos para procesar")

            successful = 0
            for i, excel_file in enumerate(sorted(excel_files), 1):
                self.logger.info(f"Procesando archivo {i}/{total_files}: {excel_file.name}")

                if self.process_single_file(excel_file, log_content):
                    successful += 1

            self.logger.info(f"\nProcesamiento completado:")
            self.logger.info(f"Total archivos: {total_files}")
            self.logger.info(f"Procesados con éxito: {successful}")
            self.logger.info(f"Fallidos: {total_files - successful}")

        except Exception as e:
            self.logger.error(f"Error en el procesamiento por lotes: {str(e)}")

class TimedBatchProcessor(BatchProcessor):
    """
    Procesador por lotes con control de tiempo de ejecución.
    
    Esta clase extiende BatchProcessor añadiendo límites de tiempo para el procesamiento:
    - 5 minutos si ya se ha encontrado una solución válida
    - 15 minutos si aún no se ha encontrado una solución válida
    """
    
    def __init__(self, data_dir: str):
        """
        Inicializa el procesador por lotes con límites de tiempo.
        
        Args:
            data_dir: Directorio que contiene los datos a procesar
        """
        super().__init__(data_dir)
        self.max_time_with_solution = 300  # 5 minutos
        self.max_time_without_solution = 900  # 15 minutos
        self.start_time = time.time()
        self.has_valid_solution = False

    def check_time_limit(self) -> bool:
        """Verifica si se ha excedido el límite de tiempo configurado."""
        elapsed_time = time.time() - self.start_time
        
        if self.has_valid_solution:
            if elapsed_time > self.max_time_with_solution:
                self.logger.info(
                    f"Proceso detenido después de {elapsed_time:.2f} segundos: "
                    "se ha alcanzado el límite de tiempo con solución válida (5 minutos)"
                )
                return True
        elif elapsed_time > self.max_time_without_solution:
            self.logger.info(
                f"Proceso detenido después de {elapsed_time:.2f} segundos: "
                "se ha alcanzado el límite máximo sin solución válida (15 minutos)"
            )
            return True
        
        return False

    def process_algorithms(self, excel_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Ejecuta los algoritmos GA y HS para un archivo dado con control de tiempo."""
        # Verificar límite de tiempo antes de empezar
        if self.check_time_limit():
            return {}, {}
            
        ga_metrics, hs_metrics = super().process_algorithms(excel_file)
        
        # Actualizar estado de solución válida
        if ga_metrics.get('fitness_final', 0) > 0 or hs_metrics.get('fitness_final', 0) > 0:
            self.has_valid_solution = True
            
        return ga_metrics, hs_metrics

    def process_all_files(self):
        """Procesa todos los archivos con control de tiempo."""
        self.start_time = time.time()
        self.logger.info("Iniciando procesamiento con límites de tiempo:")
        self.logger.info("- 5 minutos si se encuentra solución válida")
        self.logger.info("- 15 minutos si no se encuentra solución")
        
        super().process_all_files()

def find_data_directories() -> List[Path]:
    """
    Encuentra los directorios de datos disponibles.
    
    Busca el directorio 'datos_sinteticos' desde la ubicación del archivo
    batch_main.py, independientemente de dónde se encuentre ubicado.
    
    Returns:
        List[Path]: Lista de directorios de datos encontrados con timestamps válidos
        
    Raises:
        SystemExit: Si no se encuentra el directorio datos_sinteticos
    
    Autor: Juan José Jiménez González
    Universidad: Universidad Isabel I
    """
    # Buscar datos_sinteticos desde la ubicación del archivo actual
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "datos_sinteticos"
    
    if not data_dir.exists():
        print(f"Error: No se encuentra el directorio {data_dir}")
        print(f"Directorio del script: {script_dir}")
        print("Verifique que el directorio 'datos_sinteticos' existe en la misma ubicación que batch_main.py")
        sys.exit(1)
    
    # Listar directorios disponibles (solo los que contienen timestamp)
    dirs = [d for d in data_dir.iterdir() 
            if d.is_dir() and re.match(r'\d{8}-\d{6}', d.name)]
    
    if not dirs:
        print(f"Error: No se encontraron directorios con formato timestamp en {data_dir}")
        print("Los directorios deben tener formato YYYYMMDD-HHMMSS")
        sys.exit(1)
    
    return sorted(dirs)

def select_data_directory() -> str:
    """
    Solicita al usuario que seleccione un directorio de datos.
    
    Returns:
        str: Ruta al directorio seleccionado
        
    Raises:
        SystemExit: Si el usuario elige salir o si no hay directorios válidos
    """
    dirs = find_data_directories()
    
    if not dirs:
        print("Error: No se encontraron directorios de datos válidos")
        sys.exit(1)
    
    print("\nDirectorios disponibles:")
    for i, dir_path in enumerate(dirs, 1):
        excel_files = list(dir_path.glob("*.xlsx"))
        print(f"{i}. {dir_path.name} ({len(excel_files)} archivos Excel)")
    
    while True:
        try:
            choice = input("\nSeleccione el número del directorio a procesar (o 'q' para salir): ")
            if choice.lower() == 'q':
                sys.exit(0)
            
            idx = int(choice) - 1
            if 0 <= idx < len(dirs):
                selected_dir = dirs[idx]
                
                # Verificar archivos necesarios
                excel_files = list(selected_dir.glob("*.xlsx"))
                log_file = selected_dir / "log.txt"
                
                if not excel_files:
                    print(f"Error: No se encontraron archivos Excel en {selected_dir}")
                    continue
                    
                if not log_file.exists():
                    print(f"Error: No se encuentra log.txt en {selected_dir}")
                    continue
                
                print(f"\nDirectorio seleccionado: {selected_dir}")
                print(f"Archivos Excel encontrados: {len(excel_files)}")
                confirm = input("¿Desea proceder con este directorio? (s/n): ")
                
                if confirm.lower() == 's':
                    return str(selected_dir)
                
            else:
                print("Número inválido. Intente de nuevo.")
                
        except ValueError:
            print("Entrada inválida. Ingrese un número o 'q' para salir.")
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """
    Función principal del programa.
    
    Gestiona la selección del directorio de datos y la ejecución del procesamiento
    por lotes con control de tiempo. Maneja las excepciones y proporciona
    retroalimentación clara al usuario.
    """
    print("=== Procesador por Lotes de Datos Sintéticos ===")
    print("Este script procesará los datos sintéticos generados previamente")
    print("y ejecutará los algoritmos GA y HS para cada archivo.")
    print("\nLímites de tiempo configurados:")
    print("- 5 minutos si se encuentra una solución válida")
    print("- 15 minutos si no se encuentra solución válida")
    print(f"\nDirectorio raíz del proyecto: {project_root}")
    
    try:
        # Obtener directorio de datos
        data_dir = select_data_directory()
        
        # Iniciar procesamiento con control de tiempo
        processor = TimedBatchProcessor(data_dir)  # Cambiado a TimedBatchProcessor
        
        print("\nIniciando procesamiento con control de tiempo...")
        print("Presione Ctrl+C para interrumpir el proceso en cualquier momento.")
        
        processor.process_all_files()
        
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
    finally:
        print("\nProcesamiento finalizado.")

if __name__ == "__main__":
    main()    