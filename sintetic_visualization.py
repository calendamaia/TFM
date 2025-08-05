"""
Script para análisis comparativo de resultados de algoritmos AG y HS.
Permite seleccionar la carpeta de datos sintéticos a analizar.
Con ello genera un excel y gráficas comparativas3

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class ComparisonAnalyzer:
    def __init__(self):
        """Inicializa el analizador comparativo."""
        self.project_root = Path(__file__).parent
        self.synthetic_data_dir = self.project_root / "datos_sinteticos"
        self.colors = self._get_color_palette()
        self._setup_plot_style()
        
    def _get_color_palette(self) -> Dict[str, str]:
        """Retorna la paleta de colores corporativa."""
        return {
            'primary': '#E31837',      # Rojo UI1 principal
            'medium': '#FF6666',       # Rojo pastel medio
            'dark': '#CC3333',         # Rojo pastel oscuro
            'background': '#FFFFFF',    # Blanco
            'text': '#4A4A4A',         # Gris oscuro
            'grid': '#FFE6E6'          # Rojo muy claro para grilla
        }
    
    def _setup_plot_style(self):
        """Configura el estilo visual de las gráficas."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = self.colors['background']
        plt.rcParams['axes.facecolor'] = self.colors['background']
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = self.colors['grid']

    def get_available_runs(self) -> List[Path]:
        """
        Obtiene las carpetas de ejecución disponibles.
        
        Returns:
            Lista de rutas a las carpetas de ejecución ordenadas por fecha
        """
        if not self.synthetic_data_dir.exists():
            raise FileNotFoundError(f"No se encuentra el directorio {self.synthetic_data_dir}")
        
        runs = [d for d in self.synthetic_data_dir.iterdir() if d.is_dir()]
        # Ordenar por nombre (que es un datetime) de más reciente a más antiguo
        runs.sort(reverse=True)
        return runs

    def select_run(self) -> Path:
        """
        Permite al usuario seleccionar una carpeta de ejecución.
        
        Returns:
            Path a la carpeta seleccionada
            
        Raises:
            ValueError: Si no hay carpetas disponibles o la selección es inválida
        """
        runs = self.get_available_runs()
        if not runs:
            raise ValueError("No se encontraron carpetas de ejecución")
        
        print("\nCarpetas de ejecución disponibles:")
        print("-" * 50)
        for i, run in enumerate(runs, 1):
            # Intentar convertir el nombre de la carpeta a datetime para mejor visualización
            try:
                run_date = datetime.strptime(run.name, "%Y%m%d_%H%M%S")
                date_str = run_date.strftime("%d/%m/%Y %H:%M:%S")
            except ValueError:
                date_str = run.name
            print(f"{i}. {date_str}")
        
        while True:
            try:
                selection = int(input("\nSeleccione una carpeta (número): "))
                if 1 <= selection <= len(runs):
                    return runs[selection - 1]
                print("Selección fuera de rango. Intente nuevamente.")
            except ValueError:
                print("Por favor, ingrese un número válido.")

    def extract_metrics(self, log_file: Path) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Extrae métricas de tiempo y generaciones del archivo de log.
        
        Args:
            log_file: Ruta al archivo de log
            
        Returns:
            Tuple con diccionarios de tiempos y generaciones para AG y HS
        """
        times = {'AG': 0.0, 'HS': 0.0}
        generations = {'AG': 0, 'HS': 0}
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extraer métricas del AG
            ag_match = re.search(r'Métricas Algoritmo Genético:\s*\ntiempo_ejecucion:\s*([\d.]+).*?\ngeneraciones:\s*(\d+)',
                            content, re.DOTALL)
            if ag_match:
                times['AG'] = float(ag_match.group(1))
                generations['AG'] = int(ag_match.group(2))
            else:
                print(f"No se encontraron métricas AG en {log_file}")
            
            # Extraer métricas del HS
            hs_match = re.search(r'Métricas Harmony Search:\s*\ntiempo_ejecucion:\s*([\d.]+).*?\ngeneraciones:\s*(\d+)',
                            content, re.DOTALL)
            if hs_match:
                times['HS'] = float(hs_match.group(1))
                generations['HS'] = int(hs_match.group(2))
            else:
                print(f"No se encontraron métricas HS en {log_file}")
                
        except Exception as e:
            print(f"Error procesando {log_file}: {str(e)}")
        
        return times, generations

    def collect_data(self, selected_run: Path) -> pd.DataFrame:
        """
        Recolecta datos de todos los archivos de log.
        
        Args:
            selected_run: Ruta a la carpeta de ejecución seleccionada
            
        Returns:
            DataFrame con los datos recolectados
        """
        data = []
        results_dir = selected_run / "results"
        
        if not results_dir.exists():
            raise FileNotFoundError(f"No se encuentra el directorio de resultados en {selected_run}")
        
        for i in range(1, 101):  # Buscar del 001 al 100
            file_name = f"DatosGestionTribunales-{i:03d}"
            subdir = results_dir / file_name
            
            if subdir.exists():
                log_file = subdir / "processing_log.txt"
                if log_file.exists():
                    times, generations = self.extract_metrics(log_file)
                    data.append({
                        'Orden': i,
                        'Archivo': file_name,
                        'Tiempo_AG': times['AG'],
                        'Generaciones_AG': generations['AG'],
                        'Tiempo_HS': times['HS'],
                        'Generaciones_HS': generations['HS']
                    })
        
        return pd.DataFrame(data)

    def add_logo(self, fig: plt.Figure, height_ratio: float = 0.15):
        """Añade el logo de la universidad a la figura."""
        try:
            logo_path = self.project_root / "logoui1.png"
            if not logo_path.exists():
                return
            
            img = plt.imread(str(logo_path))
            height, width = img.shape[:2]
            aspect = width / height
            
            logo_height = fig.get_figheight() * height_ratio * 0.8
            logo_width = logo_height * aspect
            
            rel_height = logo_height / fig.get_figheight()
            rel_width = logo_width / fig.get_figwidth()
            
            logo_ax = fig.add_axes([0.1, 0.02, rel_width, rel_height])
            logo_ax.imshow(img)
            logo_ax.axis('off')
            
        except Exception as e:
            print(f"Error añadiendo logo: {str(e)}")

    def create_comparison_plot(self, data: pd.DataFrame, y_column: str, 
                             title: str, filename: str, output_dir: Path):
        """
        Crea una gráfica comparativa para una métrica específica.
        
        Args:
            data: DataFrame con los datos
            y_column: Columna a graficar
            title: Título de la gráfica
            filename: Nombre del archivo de salida
            output_dir: Directorio donde guardar la gráfica
        """
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        x = np.arange(len(data))
        width = 0.35
        
        # Barras
        ax.bar(x, data[y_column], width, color=self.colors['medium'], alpha=0.6)
        
        # Línea que une los valores
        ax.plot(x, data[y_column], color=self.colors['dark'], linewidth=2)
        
        # Configuración
        ax.set_title(title, pad=20, fontweight='bold', color=self.colors['text'], size=16)
        ax.set_xlabel('Número de escenario', color=self.colors['text'], size=14)
        ax.set_ylabel(y_column.replace('_', ' '), color=self.colors['text'], size=14)
        
        # Ajustes de ejes
        ax.set_xticks(x[::5])
        ax.set_xticklabels([str(i+1) for i in range(0, len(data), 5)])
        
        # Añadir logo
        self.add_logo(fig)
        
        # Guardar
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'])
        plt.close()

    def analyze_and_visualize(self):
        """Ejecuta el análisis completo y genera todas las visualizaciones."""
        try:
            # Seleccionar carpeta de ejecución
            selected_run = self.select_run()
            print(f"\nAnalizando datos de la carpeta: {selected_run.name}")
            
            # Recolectar datos
            data = self.collect_data(selected_run)
            
            # Crear directorio para resultados si no existe
            output_dir = selected_run / "analysis"
            output_dir.mkdir(exist_ok=True)
            
            # Guardar Excel
            excel_path = output_dir / "ComparacionAGyHS.xlsx"
            data.to_excel(excel_path, index=False)
            print(f"\nDatos guardados en {excel_path}")
            
            # Generar gráficas
            self.create_comparison_plot(
                data, 'Tiempo_AG', 
                'Tiempos de Ejecución - Algoritmo Genético',
                'comparacion_tiempos_AG.png',
                output_dir
            )
            
            self.create_comparison_plot(
                data, 'Tiempo_HS',
                'Tiempos de Ejecución - Harmony Search',
                'comparacion_tiempos_HS.png',
                output_dir
            )
            
            self.create_comparison_plot(
                data, 'Generaciones_AG',
                'Generaciones Necesarias - Algoritmo Genético',
                'comparacion_generaciones_AG.png',
                output_dir
            )
            
            self.create_comparison_plot(
                data, 'Generaciones_HS',
                'Generaciones Necesarias - Harmony Search',
                'comparacion_generaciones_HS.png',
                output_dir
            )
            
            print(f"\nAnálisis completado. Resultados guardados en: {output_dir}")
            
        except Exception as e:
            print(f"\nError en el análisis: {str(e)}")
        finally:
            plt.close('all')

if __name__ == "__main__":
    analyzer = ComparisonAnalyzer()
    analyzer.analyze_and_visualize()