"""
Script de análisis y visualización para comparar resultados de algoritmos de optimización.

Este módulo implementa un analizador que puede procesar tanto conjuntos de 5 archivos
como colecciones completas de datos, generando visualizaciones comparativas entre
los algoritmos AG y HS. Incluye normalización de métricas y cálculo de índice
compuesto de dimensión del problema.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
Fecha: 2024
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Constantes para colores
COLORS = {
    'AG': '#87CEEB',      # Azul cielo para AG
    'HS': '#CC3333',      # Rojo para HS
    'background': '#FFFFFF',  # Blanco
    'text': '#4A4A4A',    # Gris oscuro
    'grid': '#E5E5E5',    # Gris claro para la grilla
}

# Constantes para configuración de gráficas
PLOT_CONFIG = {
    'dpi': 300,
    'font_family': 'Arial',
    'title_size': 16,
    'label_size': 14,
    'tick_size': 12,
    'grid_alpha': 0.3,
    'scatter_size': 100,
    'scatter_alpha': 0.6,
    'line_alpha': 0.5,
}

# Constantes para dimensiones de figuras
FIGURE_DIMENSIONS = {
    'comparison': {
        'figsize': (15, 10),
        'axes_rect': [0.1, 0.25, 0.8, 0.65]
    },
    'correlation': {
        'figsize': (10, 12),
        'axes_rect': [0.1, 0.25, 0.8, 0.65]
    }
}

# Constantes para el logo
LOGO_CONFIG = {
    'height_ratio': 0.15,
    'vertical_position': 0.05,
}

class AnalysisVisualizer:
    """Clase para análisis y visualización de resultados comparativos."""
    
    def __init__(self):
        """Inicializa el analizador comparativo."""
        self.project_root = Path(__file__).parent
        self.synthetic_data_dir = self.project_root / "datos_sinteticos"
        self.colors = COLORS
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configura el estilo visual de las gráficas."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = self.colors['background']
        plt.rcParams['axes.facecolor'] = self.colors['background']
        plt.rcParams['font.family'] = PLOT_CONFIG['font_family']
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = PLOT_CONFIG['grid_alpha']
        plt.rcParams['grid.color'] = self.colors['grid']

    def create_comparison_plot(self, data: pd.DataFrame, metric_cols: List[str],
                             title: str, ylabel: str, filename: str, 
                             output_dir: Path, is_batch: bool = True):
        """Crea una gráfica comparativa para métricas específicas."""
        print(f"\nGenerando gráfica: {filename}")
        print(f"Modo: {'lote' if is_batch else 'conjunto completo'}")
        print(f"Métricas a comparar: {metric_cols}")

        try:
            # Verificar datos
            for col in metric_cols + ['composite_index']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            print("\nRango de valores:")
            for col in metric_cols + ['composite_index']:
                print(f"{col}: [{data[col].min():.2f}, {data[col].max():.2f}]")

            # Crear figura
            dims = FIGURE_DIMENSIONS['comparison']
            fig = plt.figure(figsize=dims['figsize'])
            ax = fig.add_axes(dims['axes_rect'])

            composite_unique = data['composite_index'].nunique()
            x = range(len(data)) if composite_unique <= 1 else data['composite_index'].values

            if not is_batch:
                # Gráfico de dispersión
                for i, metric in enumerate(metric_cols):
                    color = self.colors['AG'] if i == 0 else self.colors['HS']
                    label = 'Algoritmo Genético' if i == 0 else 'Harmony Search'
                    
                    ax.scatter(x, data[metric], 
                             label=label, 
                             color=color, 
                             alpha=PLOT_CONFIG['scatter_alpha'],
                             s=PLOT_CONFIG['scatter_size'])
                    
                    if composite_unique > 1:
                        try:
                            z = np.polyfit(x, data[metric], 1)
                            p = np.poly1d(z)
                            x_sorted = np.sort(x)
                            ax.plot(x_sorted, p(x_sorted), '--', 
                                  color=color, 
                                  alpha=PLOT_CONFIG['line_alpha'])
                        except Exception as e:
                            print(f"No se pudo generar línea de tendencia para {metric}: {e}")
            else:
                # Gráfico de barras agrupadas
                width = 0.35
                x_pos = np.arange(len(data))
                
                bars1 = ax.bar(x_pos - width/2, data[metric_cols[0]], width,
                             label='Algoritmo Genético', color=self.colors['AG'])
                bars2 = ax.bar(x_pos + width/2, data[metric_cols[1]], width,
                             label='Harmony Search', color=self.colors['HS'])
                
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom')
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'Caso {i+1}\n(Dim: {data.iloc[i]["composite_index"]:.2f})'
                                   for i in range(len(data))], rotation=45)

            # Configurar gráfica
            ax.set_title(title, pad=20, fontweight='bold', 
                        color=self.colors['text'], 
                        size=PLOT_CONFIG['title_size'])
            ax.set_xlabel('Número de caso' if composite_unique <= 1 else 'Índice de Dimensión del Problema',
                         color=self.colors['text'], 
                         size=PLOT_CONFIG['label_size'])
            ax.set_ylabel(ylabel, color=self.colors['text'], 
                         size=PLOT_CONFIG['label_size'])

            # Ajustar ejes
            if not is_batch and composite_unique > 1:
                ax.set_xlim(min(x) * 0.9, max(x) * 1.1)
            y_max = max(data[metric_cols].max().max() * 1.1, 0.1)
            y_min = min(data[metric_cols].min().min() * 0.9, 0)
            ax.set_ylim(y_min, y_max)

            # Configuración adicional
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1))
            
            # Añadir logo y guardar
            self._add_logo(fig)
            plt.savefig(output_dir / filename, 
                       dpi=PLOT_CONFIG['dpi'], 
                       bbox_inches='tight',
                       facecolor=self.colors['background'])
            print(f"Gráfica guardada en: {output_dir / filename}")
            plt.close()

        except Exception as e:
            print(f"\nError generando gráfica {filename}:")
            print(f"Tipo de error: {type(e).__name__}")
            print(f"Mensaje de error: {str(e)}")
            plt.close()
            raise



    
   
    def run(self):
        """
        Ejecuta el análisis según la selección del usuario.
        Permite elegir entre analizar un lote de 5 archivos o el conjunto completo.
        """
        try:
            print("=== Analizador de Resultados de Optimización ===")
            print("\nOpciones disponibles:")
            print("1. Analizar lote de 5 archivos")
            print("2. Analizar conjunto completo")
            
            option = input("\nSeleccione una opción (1/2): ").strip()
            
            if option not in ['1', '2']:
                raise ValueError("Opción no válida")
            
            # Obtener directorios disponibles
            runs = self.get_available_runs()
            if not runs:
                raise ValueError("No se encontraron directorios de datos para analizar")
            
            print("\nDirectorios disponibles:")
            for i, run in enumerate(runs, 1):
                try:
                    # Extraer fecha del nombre del directorio y formatearla
                    run_date = datetime.strptime(run.name.split('-')[0], '%Y%m%d')
                    date_str = run_date.strftime("%d/%m/%Y")
                    print(f"{i}. {date_str} - {run.name}")
                except ValueError:
                    # Si no se puede extraer la fecha, mostrar solo el nombre
                    print(f"{i}. {run.name}")
            
            # Selección del directorio
            while True:
                try:
                    selection = int(input("\nSeleccione el número del directorio a analizar: "))
                    if 1 <= selection <= len(runs):
                        selected_dir = runs[selection - 1]
                        break
                    print("Selección fuera de rango")
                except ValueError:
                    print("Por favor, ingrese un número válido")
            
            # Ejecutar análisis según la opción seleccionada
            if option == '1':
                self.analyze_batch(selected_dir)
            else:
                self.analyze_complete_set(selected_dir)
            
        except Exception as e:
            print(f"\nError durante el análisis: {str(e)}")
        finally:
            plt.close('all')
        
    def _get_color_palette(self) -> Dict[str, str]:
        """Retorna la paleta de colores corporativa de la UI1."""
        return {
            'primary': '#E31837',      # Rojo UI1 principal
            'medium': '#FF6666',       # Rojo pastel medio
            'dark': '#CC3333',         # Rojo pastel oscuro
            'background': '#FFFFFF',    # Blanco
            'text': '#4A4A4A',         # Gris oscuro
            'grid': '#FFE6E6'          # Rojo muy claro para grilla
        }
    
    
    def calculate_composite_index(self, info_data: pd.DataFrame) -> float:
        """
        Calcula el índice compuesto de dimensión del problema.
        
        El índice se calcula como: √((alumnos × profesores) × (aulas × slots_temporales))
        Este enfoque equilibra la influencia de:
        - Complejidad de asignación (alumnos × profesores)
        - Complejidad espacial (aulas × slots_temporales)
        """
        try:
            # Primero intentar obtener datos de las columnas directamente
            if all(col in info_data.columns for col in ['num_students', 'num_professors', 'num_buildings', 'num_aulas']):
                num_students = info_data['num_students'].iloc[0]
                num_professors = info_data['num_professors'].iloc[0]
                num_buildings = info_data['num_buildings'].iloc[0]
                num_aulas = info_data['num_aulas'].iloc[0]
            # Si no funciona, intentar con el formato Parámetro/Valor
            elif all(col in info_data.columns for col in ['Parámetro', 'Valor']):
                info_dict = dict(zip(info_data['Parámetro'], info_data['Valor']))
                num_students = info_dict.get('num_students', 0)
                num_professors = info_dict.get('num_professors', 0)
                num_buildings = info_dict.get('num_buildings', 1)
                num_aulas = info_dict.get('num_aulas', 1)
            else:
                # Si no encontramos los datos en ningún formato conocido, usar los primeros valores
                num_students = len(info_data)
                num_professors = max(3, num_students // 3)  # Al menos 3 profesores
                num_buildings = 1
                num_aulas = 1
            
            # Asegurar que todos los valores sean al menos 1
            num_students = max(1, num_students)
            num_professors = max(3, num_professors)
            num_buildings = max(1, num_buildings)
            num_aulas = max(1, num_aulas)
            
            print(f"Dimensiones detectadas:")
            print(f"- Estudiantes: {num_students}")
            print(f"- Profesores: {num_professors}")
            print(f"- Edificios: {num_buildings}")
            print(f"- Aulas: {num_aulas}")
            
            # Calcular slots temporales (10 slots por día, considerando mañana y tarde)
            slots_per_day = 10
            total_slots = num_buildings * num_aulas * slots_per_day
            
            # Calcular índice compuesto
            assignment_complexity = num_students * num_professors
            spatial_complexity = num_aulas * total_slots
            
            composite_index = np.sqrt(assignment_complexity * spatial_complexity)
            
            print(f"- Total slots: {total_slots}")
            print(f"- Índice compuesto: {composite_index:.2f}")
            
            return composite_index
            
        except Exception as e:
            print(f"Error calculando índice compuesto: {str(e)}")
            return 1.0  # Valor por defecto en caso de error

    def extract_metrics(self, log_file: Path) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Extrae métricas y datos de información del archivo de log.
        
        Args:
            log_file: Ruta al archivo de log
            
        Returns:
            Tupla con diccionario de métricas y DataFrame de información
        """
        try:
            # Extraer métricas del log
            metrics = self._process_log_file(log_file)
            
            # Leer archivo Excel asociado
            excel_files = [f for f in os.listdir(log_file.parent) if f.endswith('.xlsx')]
            if not excel_files:
                raise FileNotFoundError("No se encontró archivo Excel asociado")
            
            excel_path = log_file.parent / excel_files[0]
            
            # Intentar leer la hoja INFO primero
            try:
                info_sheet = pd.read_excel(excel_path, sheet_name='INFO')
            except:
                # Si no existe la hoja INFO, extraer información de otras hojas
                alumnos_df = pd.read_excel(excel_path, sheet_name='Disponibilidad-alumnos-turnos')
                tribunal_df = pd.read_excel(excel_path, sheet_name='Disponibilidad-tribunal-turnos')
                turnos_df = pd.read_excel(excel_path, sheet_name='Turnos')
                
                # Extraer información dimensional
                num_students = len(alumnos_df) - 1  # -1 por la fila de encabezado
                num_professors = len(tribunal_df) - 1  # -1 por la fila de encabezado
                
                # Extraer información de edificios y aulas de los turnos
                edificios = set()
                aulas = set()
                for turno in turnos_df['Turno']:
                    partes = turno.split('-')
                    if len(partes) >= 3:
                        if partes[-1].startswith('E'):
                            edificios.add(partes[-1])
                        for parte in partes:
                            if parte.startswith('A'):
                                aulas.add(parte)
                
                num_buildings = max(1, len(edificios))
                num_aulas = max(1, len(aulas))
                
                # Crear DataFrame de información
                info_sheet = pd.DataFrame({
                    'num_students': [num_students],
                    'num_professors': [num_professors],
                    'num_buildings': [num_buildings],
                    'num_aulas': [num_aulas]
                })
            
            return metrics, info_sheet
            
        except Exception as e:
            print(f"Error procesando {log_file}: {str(e)}")
            # Retornar valores por defecto y un DataFrame vacío pero con las columnas necesarias
            return {
                'AG_time': 0.0, 'HS_time': 0.0,
                'AG_generations': 0, 'HS_generations': 0,
                'AG_fitness': 0.0, 'HS_fitness': 0.0
            }, pd.DataFrame({
                'num_students': [1],
                'num_professors': [3],
                'num_buildings': [1],
                'num_aulas': [1]
            })

    def get_available_runs(self) -> List[Path]:
        """
        Obtiene las carpetas de ejecución disponibles, agrupadas por timestamp base.
        
        Returns:
            List[Path]: Lista de directorios base únicos
        """
        if not self.synthetic_data_dir.exists():
            raise FileNotFoundError(f"No se encuentra el directorio {self.synthetic_data_dir}")
        
        # Encontrar todos los directorios que siguen el patrón
        all_dirs = [d for d in self.synthetic_data_dir.iterdir() 
                    if d.is_dir() and re.match(r'\d{8}-\d{6}', d.name)]
        
        # Extraer timestamps base únicos
        base_timestamps = set()
        for d in all_dirs:
            # Extraer el timestamp base (YYYYMMDD-HHMMSS)
            base_match = re.match(r'(\d{8}-\d{6})', d.name)
            if base_match:
                base_timestamps.add(base_match.group(1))
        
        # Crear Path objects para los timestamps base
        base_dirs = [self.synthetic_data_dir / timestamp for timestamp in sorted(base_timestamps, reverse=True)]
        return base_dirs

    def get_all_subdirs(self, base_dir: Path) -> List[Path]:
        """
        Obtiene todos los subdirectorios asociados a un timestamp base.
        
        Args:
            base_dir: Directorio base con el timestamp
            
        Returns:
            List[Path]: Lista de todos los subdirectorios asociados
        """
        base_pattern = base_dir.name + r'-\d+-\d+'
        all_subdirs = [d for d in self.synthetic_data_dir.iterdir() 
                    if d.is_dir() and re.match(base_pattern, d.name)]
        
        # Ordenar por el número de inicio del rango
        return sorted(all_subdirs, 
                    key=lambda x: int(re.search(r'-(\d+)-\d+', x.name).group(1)))
    
    def _add_logo(self, fig: plt.Figure):
        """Añade el logo de la universidad a la figura."""
        try:
            logo_path = self.project_root / "logoui1.png"
            if logo_path.exists():
                img = plt.imread(str(logo_path))
                height, width = img.shape[:2]
                aspect = width / height
                
                logo_height = fig.get_figheight() * 0.15 * 0.8
                logo_width = logo_height * aspect
                
                rel_height = logo_height / fig.get_figheight()
                rel_width = logo_width / fig.get_figwidth()
                
                logo_ax = fig.add_axes([0.1, 0.02, rel_width, rel_height])
                logo_ax.imshow(img)
                logo_ax.axis('off')
        except Exception as e:
            print(f"Error añadiendo logo: {str(e)}")

    def analyze_batch(self, selected_dir: Path):
        """Analiza un conjunto de 5 archivos."""
        print(f"\nAnalizando lote en: {selected_dir}")
        
        # Crear directorio para resultados
        output_dir = selected_dir / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Recolectar datos
        print("\nRecolectando datos...")
        data = []
        results_dir = selected_dir / "results"
        
        if not results_dir.exists():
            raise FileNotFoundError(f"No se encuentra el directorio de resultados en {selected_dir}")
        
        # Procesar cada subdirectorio en orden
        subdirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
        
        for subdir in subdirs:
            # Buscar archivos
            log_files = list(subdir.glob("processing_log.txt"))
            excel_files = list(subdir.glob("DatosGestionTribunales*.xlsx"))
            
            print(f"\nProcesando directorio: {subdir.name}")
            print(f"Archivos encontrados - Logs: {len(log_files)}, Excel: {len(excel_files)}")
            
            if log_files and excel_files:
                try:
                    # Procesar log
                    metrics = self._process_log_file(log_files[0])
                    print("\nMétricas extraídas del log:")
                    for k, v in metrics.items():
                        print(f"{k}: {v}")
                    
                    # Extraer dimensiones del Excel
                    excel_data = self._extract_dimensions_from_excel(excel_files[0])
                    composite_index = self.calculate_composite_index(excel_data)
                    print(f"\nÍndice compuesto calculado: {composite_index}")
                    
                    entry = {
                        'filename': subdir.name,
                        'composite_index': composite_index,
                        **metrics
                    }
                    data.append(entry)
                    
                except Exception as e:
                    print(f"Error procesando {subdir.name}: {str(e)}")
                    print("Continuando con el siguiente archivo...")
        
        if not data:
            raise ValueError("No se encontraron datos para analizar")
        
        # Crear DataFrame y generar visualizaciones
        df = pd.DataFrame(data)
        print("\nDatos recolectados:")
        print(df)
        
        # Asegurar que todos los valores son numéricos
        for col in df.columns:
            if col != 'filename':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\nGenerando visualizaciones...")
        
        # Gráficas comparativas
        metrics_to_plot = [
            (['AG_time', 'HS_time'], 'Comparación de Tiempos de Ejecución', 'Tiempo (segundos)'),
            (['AG_generations', 'HS_generations'], 'Comparación de Generaciones', 'Número de Generaciones'),
            (['AG_fitness', 'HS_fitness'], 'Comparación de Fitness', 'Valor de Fitness')
        ]
        
        for metrics_pair, title, ylabel in metrics_to_plot:
            try:
                self.create_comparison_plot(
                    df, metrics_pair, title, ylabel,
                    f'comparacion_{metrics_pair[0].lower()}.png',
                    output_dir
                )
            except Exception as e:
                print(f"Error generando gráfica {title}: {str(e)}")
        
        # Guardar datos en Excel
        excel_path = output_dir / 'analisis_comparativo.xlsx'
        df.to_excel(excel_path, index=False)
        print(f"\nResultados guardados en: {output_dir}")

    def _extract_dimensions_from_excel(self, excel_file: Path) -> pd.DataFrame:
        """
        Extrae las dimensiones del problema directamente de las hojas del Excel.
        """
        try:
            # Leer las hojas relevantes
            alumnos_df = pd.read_excel(excel_file, sheet_name='Disponibilidad-alumnos-turnos')
            tribunal_df = pd.read_excel(excel_file, sheet_name='Disponibilidad-tribunal-turnos')
            turnos_df = pd.read_excel(excel_file, sheet_name='Turnos')
            
            # Extraer dimensiones
            num_students = len(alumnos_df) - 1  # -1 por la fila de encabezado
            num_professors = len(tribunal_df) - 1
            
            # Extraer información de edificios y aulas de los turnos
            edificios = set()
            aulas = set()
            for turno in turnos_df['Turno']:
                partes = turno.split('-')
                if len(partes) >= 3:
                    for parte in partes:
                        if parte.startswith('E'):
                            edificios.add(parte)
                        elif parte.startswith('A'):
                            aulas.add(parte)
            
            num_buildings = max(1, len(edificios))
            num_aulas = max(1, len(aulas))
            
            # Crear DataFrame
            return pd.DataFrame({
                'Parámetro': ['num_students', 'num_professors', 'num_buildings', 'num_aulas'],
                'Valor': [num_students, num_professors, num_buildings, num_aulas]
            })
            
        except Exception as e:
            print(f"Error extrayendo dimensiones de {excel_file}: {str(e)}")
            # Retornar valores por defecto si hay error
            return pd.DataFrame({
                'Parámetro': ['num_students', 'num_professors', 'num_buildings', 'num_aulas'],
                'Valor': [1, 3, 1, 1]
            })
    
    def analyze_complete_set(self, base_dir: Path):
        """
        Analiza el conjunto completo de datos para un timestamp base.
        
        Args:
            base_dir: Directorio base que contiene el timestamp
        """
        print(f"\nAnalizando conjunto completo para timestamp: {base_dir.name}")
        
        # Obtener todos los subdirectorios asociados
        all_subdirs = self.get_all_subdirs(base_dir)
        if not all_subdirs:
            raise ValueError(f"No se encontraron subdirectorios para {base_dir.name}")
        
        print(f"\nSubdirectorios encontrados: {len(all_subdirs)}")
        for subdir in all_subdirs:
            print(f"- {subdir.name}")
        
        # Crear directorio para resultados
        output_dir = self.synthetic_data_dir / base_dir.name / "analysis_complete"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Recolectar datos de todos los subdirectorios
        data = []
        
        for subdir in all_subdirs:
            results_dir = subdir / "results"
            if not results_dir.exists():
                print(f"No se encuentra directorio results en {subdir}")
                continue
                
            for case_dir in results_dir.glob("DatosGestionTribunales*"):
                if case_dir.is_dir():
                    log_file = case_dir / "processing_log.txt"
                    if log_file.exists():
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Extraer métricas
                            # Sección Algoritmo Genético
                            ag_section = content.split("Harmony Search:")[0] if "Harmony Search:" in content else content
                            hs_section = content.split("Harmony Search:")[1] if "Harmony Search:" in content else ""

                            metrics_match = {
                                'AG_time': re.search(r'Tiempo de ejecución:\s*([\d.]+)', ag_section),
                                'AG_gens': re.search(r'Generaciones:\s*(\d+)', ag_section),
                                'AG_fitness': re.search(r'Valor absoluto:\s*([\d.]+)', ag_section),
                                'HS_time': re.search(r'Tiempo de ejecución:\s*([\d.]+)', hs_section),
                                'HS_gens': re.search(r'Generaciones:\s*(\d+)', hs_section),
                                'HS_fitness': re.search(r'Valor absoluto:\s*([\d.]+)', hs_section)
                            }
                            
                            # Extraer número de caso del nombre del directorio
                            case_num = int(re.search(r'(\d+)', case_dir.name).group(1))
                            
                            entry = {
                                'filename': case_dir.name,
                                'case_number': case_num,
                                'AG_time': float(metrics_match['AG_time'].group(1)) if metrics_match['AG_time'] else 0,
                                'HS_time': float(metrics_match['HS_time'].group(1)) if metrics_match['HS_time'] else 0,
                                'AG_generations': int(metrics_match['AG_gens'].group(1)) if metrics_match['AG_gens'] else 0,
                                'HS_generations': int(metrics_match['HS_gens'].group(1)) if metrics_match['HS_gens'] else 0,
                                'AG_fitness': float(metrics_match['AG_fitness'].group(1)) if metrics_match['AG_fitness'] else 0,
                                'HS_fitness': float(metrics_match['HS_fitness'].group(1)) if metrics_match['HS_fitness'] else 0
                            }
                            
                            # Leer dimensiones del problema
                            excel_files = list(case_dir.glob("*.xlsx"))
                            if excel_files:
                                try:
                                    info_data = pd.read_excel(excel_files[0], sheet_name='INFO')
                                    entry['composite_index'] = self.calculate_composite_index(info_data)
                                except Exception as e:
                                    print(f"Error leyendo dimensiones: {str(e)}")
                                    entry['composite_index'] = 1.0
                            
                            data.append(entry)
                            
                        except Exception as e:
                            print(f"Error procesando {log_file}: {str(e)}")
                            continue
        
        if not data:
            raise ValueError("No se encontraron datos para analizar")
        
        # Crear DataFrame y ordenar por número de caso
        df = pd.DataFrame(data)
        df = df.sort_values('case_number')
        
        # Generar visualizaciones
        print(f"\nGenerando visualizaciones para {len(data)} casos...")
        
        self.create_comparison_plot(
            df, ['AG_time', 'HS_time'],
            'Análisis de Tiempos de Ejecución - Conjunto Completo',
            'Tiempo (segundos)',
            'analisis_tiempos_completo.png',
            output_dir,
            is_batch=False
        )
        
        self.create_comparison_plot(
            df, ['AG_generations', 'HS_generations'],
            'Análisis de Generaciones Necesarias - Conjunto Completo',
            'Número de Generaciones',
            'analisis_generaciones_completo.png',
            output_dir,
            is_batch=False
        )
        
        self.create_comparison_plot(
            df, ['AG_fitness', 'HS_fitness'],
            'Análisis de Fitness Alcanzado - Conjunto Completo',
            'Valor de Fitness',
            'analisis_fitness_completo.png',
            output_dir,
            is_batch=False
        )
        
        # Generar gráficas adicionales y análisis estadístico
        self._create_correlation_plots(df, output_dir)
        self._save_statistical_analysis(df, output_dir)
        
        print(f"\nAnálisis del conjunto completo finalizado.")
        print(f"Resultados guardados en: {output_dir}")

    def _create_correlation_plots(self, df: pd.DataFrame, output_dir: Path):
        """Genera gráficas de correlación entre métricas de ambos algoritmos."""
        metrics_pairs = [
            ('AG_time', 'HS_time', 'Correlación de Tiempos de Ejecución'),
            ('AG_generations', 'HS_generations', 'Correlación de Generaciones'),
            ('AG_fitness', 'HS_fitness', 'Correlación de Fitness')
        ]
        
        for metric1, metric2, title in metrics_pairs:
            try:
                dims = FIGURE_DIMENSIONS['correlation']
                fig = plt.figure(figsize=dims['figsize'])
                ax = fig.add_axes(dims['axes_rect'])
                
                # Crear gráfico de dispersión
                ax.scatter(df[metric1], df[metric2], 
                        color='#4A4A4A',  # Color neutro para los puntos
                        alpha=PLOT_CONFIG['scatter_alpha'],
                        label='Casos')
                
                # Añadir línea de referencia y=x
                min_val = min(df[metric1].min(), df[metric2].min())
                max_val = max(df[metric1].max(), df[metric2].max())
                ax.plot([min_val, max_val], [min_val, max_val], 
                    '--', color=self.colors['HS'], 
                    alpha=PLOT_CONFIG['line_alpha'])
                
                # Añadir línea de tendencia
                try:
                    z = np.polyfit(df[metric1], df[metric2], 1)
                    p = np.poly1d(z)
                    x_sorted = sorted(df[metric1])
                    ax.plot(x_sorted, p(x_sorted), 
                        '-', color=self.colors['text'], 
                        alpha=PLOT_CONFIG['line_alpha'])
                except Exception as e:
                    print(f"No se pudo generar línea de tendencia: {str(e)}")
                
                # Configuración de etiquetas y título
                ax.set_xlabel(f'Algoritmo Genético - {metric1.split("_")[1]}',
                            color=self.colors['text'], 
                            size=PLOT_CONFIG['label_size'])
                ax.set_ylabel(f'Harmony Search - {metric2.split("_")[1]}',
                            color=self.colors['text'], 
                            size=PLOT_CONFIG['label_size'])
                ax.set_title(title, pad=20, fontweight='bold', 
                            color=self.colors['text'], 
                            size=PLOT_CONFIG['title_size'])
                
                # Añadir logo y guardar
                self._add_logo(fig)
                output_path = output_dir / f'correlacion_{metric1.lower()}.png'
                plt.savefig(output_path, 
                        dpi=PLOT_CONFIG['dpi'], 
                        bbox_inches='tight',
                        facecolor=self.colors['background'])
                plt.close()
                
            except Exception as e:
                print(f"Error generando gráfica de correlación {title}: {str(e)}")
                plt.close()
    
    def _save_statistical_analysis(self, df: pd.DataFrame, output_dir: Path):
        """
        Guarda análisis estadístico detallado en Excel.
        """
        try:
            with pd.ExcelWriter(output_dir / 'analisis_completo.xlsx') as writer:
                # Datos originales
                df.to_excel(writer, sheet_name='Datos_Completos', index=False)
                
                # Estadísticas descriptivas
                stats_df = df[[
                    'composite_index', 'AG_time', 'HS_time',
                    'AG_generations', 'HS_generations',
                    'AG_fitness', 'HS_fitness'
                ]].describe()
                stats_df.to_excel(writer, sheet_name='Estadisticas')
                
                # Análisis comparativo
                comparison_data = {
                    'Métrica': [
                        'Tiempo promedio (s)',
                        'Generaciones promedio',
                        'Fitness promedio',
                        'Tiempo máximo (s)',
                        'Generaciones máximas',
                        'Fitness máximo',
                        'Ratio tiempo AG/HS',
                        'Ratio generaciones AG/HS',
                        'Ratio fitness AG/HS'
                    ],
                    'Algoritmo Genético': [
                        df['AG_time'].mean(),
                        df['AG_generations'].mean(),
                        df['AG_fitness'].mean(),
                        df['AG_time'].max(),
                        df['AG_generations'].max(),
                        df['AG_fitness'].max(),
                        df['AG_time'].mean() / df['HS_time'].mean(),
                        df['AG_generations'].mean() / df['HS_generations'].mean(),
                        df['AG_fitness'].mean() / df['HS_fitness'].mean()
                    ],
                    'Harmony Search': [
                        df['HS_time'].mean(),
                        df['HS_generations'].mean(),
                        df['HS_fitness'].mean(),
                        df['HS_time'].max(),
                        df['HS_generations'].max(),
                        df['HS_fitness'].max(),
                        '-',
                        '-',
                        '-'
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Comparativa', index=False)
                
                # Solo crear análisis por dimensión si hay más de un valor único
                if df['composite_index'].nunique() > 1:
                    dimension_ranges = pd.qcut(df['composite_index'], 4, 
                                            labels=['Pequeño', 'Mediano', 
                                                    'Grande', 'Muy Grande'])
                    dimension_analysis = df.groupby(dimension_ranges).agg({
                        'AG_time': ['mean', 'std', 'min', 'max'],
                        'HS_time': ['mean', 'std', 'min', 'max'],
                        'AG_generations': ['mean', 'std', 'min', 'max'],
                        'HS_generations': ['mean', 'std', 'min', 'max'],
                        'AG_fitness': ['mean', 'std', 'min', 'max'],
                        'HS_fitness': ['mean', 'std', 'min', 'max']
                    }).round(2)
                    dimension_analysis.to_excel(writer, sheet_name='Analisis_Dimension')
                
        except Exception as e:
            print(f"Error guardando análisis estadístico: {str(e)}")

    def _process_log_file(self, log_file: Path) -> Dict[str, float]:
        """
        Procesa un archivo de log para extraer las métricas de rendimiento.
        
        Args:
            log_file: Ruta al archivo de log
            
        Returns:
            Diccionario con las métricas extraídas
        """
        metrics = {
            'AG_time': 0.0, 'HS_time': 0.0,
            'AG_generations': 0, 'HS_generations': 0,
            'AG_fitness': 0.0, 'HS_fitness': 0.0
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\nProcesando archivo: {log_file.name}")
            
            # Separar contenido en secciones de AG y HS
            sections = content.split("Harmony Search:")
            ag_section = sections[0]
            hs_section = sections[1] if len(sections) > 1 else ""
            
            # Procesar sección AG
            ag_time_match = re.search(r'Tiempo de ejecución:\s*([\d.]+)\s*segundos', ag_section)
            ag_gens_match = re.search(r'Generaciones:\s*(\d+)', ag_section)
            ag_fitness_match = re.search(r'Valor absoluto:\s*([\d.]+)', ag_section)
            
            if ag_time_match:
                metrics['AG_time'] = float(ag_time_match.group(1))
                print(f"AG tiempo: {metrics['AG_time']:.2f} segundos")
            
            if ag_gens_match:
                metrics['AG_generations'] = int(ag_gens_match.group(1))
                print(f"AG generaciones: {metrics['AG_generations']}")
            
            if ag_fitness_match:
                metrics['AG_fitness'] = float(ag_fitness_match.group(1))
                print(f"AG fitness: {metrics['AG_fitness']:.4f}")
            
            # Procesar sección HS
            if hs_section:
                hs_time_match = re.search(r'Tiempo de ejecución:\s*([\d.]+)\s*segundos', hs_section)
                hs_gens_match = re.search(r'Generaciones:\s*(\d+)', hs_section)
                hs_fitness_match = re.search(r'Valor absoluto:\s*([\d.]+)', hs_section)
                
                if hs_time_match:
                    metrics['HS_time'] = float(hs_time_match.group(1))
                    print(f"HS tiempo: {metrics['HS_time']:.2f} segundos")
                
                if hs_gens_match:
                    metrics['HS_generations'] = int(hs_gens_match.group(1))
                    print(f"HS generaciones: {metrics['HS_generations']}")
                
                if hs_fitness_match:
                    metrics['HS_fitness'] = float(hs_fitness_match.group(1))
                    print(f"HS fitness: {metrics['HS_fitness']:.4f}")
            
            # Verificar si se encontraron métricas
            metrics_found = sum(1 for v in metrics.values() if v != 0)
            if metrics_found == 0:
                print("\n¡Advertencia! No se encontraron métricas en el archivo")
                print("\nContenido del archivo (primeros 1000 caracteres):")
                print("-" * 80)
                print(content[:1000])
                print("-" * 80)
                
                # Intentar con patrones alternativos
                alt_patterns = {
                    'AG_time': r'Tiempo AG:\s*([\d.]+)',
                    'AG_generations': r'Generaciones AG:\s*(\d+)',
                    'AG_fitness': r'Fitness AG:\s*([\d.]+)',
                    'HS_time': r'Tiempo HS:\s*([\d.]+)',
                    'HS_generations': r'Generaciones HS:\s*(\d+)',
                    'HS_fitness': r'Fitness HS:\s*([\d.]+)'
                }
                
                print("\nBuscando con patrones alternativos...")
                for metric, pattern in alt_patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        value = float(match.group(1)) if 'time' in metric or 'fitness' in metric else int(match.group(1))
                        metrics[metric] = value
                        print(f"Encontrado {metric}: {value}")
            else:
                print(f"\nMétricas encontradas: {metrics_found} de 6")
            
            return metrics
            
        except Exception as e:
            print(f"Error procesando archivo de log {log_file}: {str(e)}")
            import traceback
            print("Stacktrace:")
            print(traceback.format_exc())
            return metrics
    
def main():
    """Función principal del programa."""
    try:
        analyzer = AnalysisVisualizer()
        analyzer.run()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("\nPrograma finalizado.")

if __name__ == "__main__":
    main()