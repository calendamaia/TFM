"""
Módulo de visualización para el análisis comparativo de algoritmos de optimización.

Este módulo proporciona funciones para generar visualizaciones profesionales
de los resultados obtenidos por los algoritmos de optimización.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Tuple

def get_red_palette() -> Dict[str, str]:
    """
    Retorna la paleta de colores corporativa de la UI1.
    
    Returns:
        Dict[str, str]: Diccionario con los códigos de color
    """
    return {
        'primary': '#E31837',      # Rojo UI1 principal
        'light': '#FF9999',        # Rojo pastel claro
        'medium': '#FF6666',       # Rojo pastel medio
        'dark': '#CC3333',         # Rojo pastel oscuro
        'accent': '#FFB3B3',       # Rojo pastel muy claro
        'background': '#FFFFFF',    # Blanco
        'text': '#4A4A4A',         # Gris oscuro
        'grid': '#FFE6E6'          # Rojo muy claro para la grilla
    }

def set_custom_style():
    """
    Configura el estilo visual personalizado para las gráficas.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = get_red_palette()
    
    plt.rcParams['figure.facecolor'] = colors['background']
    plt.rcParams['axes.facecolor'] = colors['background']
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = colors['grid']

def add_logo_and_legend(fig: plt.Figure, ax: plt.Axes, logo_path: str, legend_elements: List[Any], height_ratio: float = 0.15):
    """
    Añade el logo y la leyenda a una figura.
    """
    try:
        # Crear eje para el área inferior
        bottom_ax = fig.add_axes([0.1, 0.02, 0.8, height_ratio], zorder=-1)
        bottom_ax.axis('off')
        
        # Cargar y dimensionar el logo
        img = plt.imread(logo_path)
        height, width = img.shape[:2]
        aspect = width / height
        
        logo_height = fig.get_figheight() * height_ratio * 0.8
        logo_width = logo_height * aspect
        
        rel_height = logo_height / fig.get_figheight()
        rel_width = logo_width / fig.get_figwidth()
        
        # Añadir logo
        logo_ax = fig.add_axes([0.1, 0.02, rel_width, rel_height], zorder=-1)
        logo_ax.imshow(img)
        logo_ax.axis('off')
        
        # Añadir leyenda si hay elementos
        if legend_elements:
            legend_ax = fig.add_axes([0.1 + rel_width + 0.05, 0.02, 0.8 - rel_width - 0.15, rel_height])
            legend_ax.axis('off')
            legend_ax.legend(handles=legend_elements, loc='center left', 
                           ncol=len(legend_elements), frameon=False,
                           fontsize=14)
            
    except Exception as e:
        print(f"No se pudo cargar el logo: {str(e)}")

def safe_normalize(fitness_history: List[float]) -> List[float]:
    """
    Normaliza una lista de valores de fitness de manera segura.
    """
    if not fitness_history:
        return []
    
    fitness_min = min(fitness_history)
    fitness_max = max(fitness_history)
    
    if abs(fitness_max - fitness_min) < 1e-10:
        return [1.0] * len(fitness_history)
    
    return [(f - fitness_min) / (fitness_max - fitness_min) for f in fitness_history]

def calculate_improvements(fitness_history: List[float]) -> List[float]:
    """
    Calcula las mejoras entre valores consecutivos de fitness.
    """
    return [fitness_history[i] - fitness_history[i-1] for i in range(1, len(fitness_history))]

def generate_metric_plot(ax: plt.Axes, ga_value: float, hs_value: float, 
                        title: str, ylabel: str, colors: Dict[str, str]):
    """
    Genera una gráfica de barras comparativa para una métrica específica.
    
    Args:
        ax (plt.Axes): Ejes donde generar la gráfica
        ga_value (float): Valor para el algoritmo genético
        hs_value (float): Valor para harmony search
        title (str): Título de la gráfica
        ylabel (str): Etiqueta del eje Y
        colors (Dict[str, str]): Paleta de colores a utilizar
    """
    width = 0.35
    x = np.arange(2)
    
    bars = ax.bar(x, [ga_value, hs_value], width, 
                 color=[colors['medium'], colors['dark']])
    
    for bar in bars:
        height = bar.get_height()
        if height >= 100:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
    
    ax.set_title(title, pad=20, fontweight='bold', color=colors['text'], fontsize=16)
    ax.set_ylabel(ylabel, color=colors['text'], fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['AG', 'HS'])
    
    ymax = max(ga_value, hs_value)
    ax.set_ylim(0, ymax * 1.15)

def generate_fitness_evolution_plot(ga_fitness_history: List[float], 
                                 hs_fitness_history: List[float],
                                 ax: plt.Axes,
                                 colors: Dict[str, str]) -> List[plt.Line2D]:
    """
    Genera una gráfica de evolución del fitness para ambos algoritmos.
    
    Args:
        ga_fitness_history (List[float]): Historial de fitness del AG
        hs_fitness_history (List[float]): Historial de fitness del HS
        ax (plt.Axes): Ejes donde generar la gráfica
        colors (Dict[str, str]): Paleta de colores a utilizar
        
    Returns:
        List[plt.Line2D]: Elementos de la leyenda
    """
    ga_constant = len(set(ga_fitness_history)) == 1
    hs_constant = len(set(hs_fitness_history)) == 1
    ga_value = ga_fitness_history[0] if ga_constant else None
    hs_value = hs_fitness_history[0] if hs_constant else None
    
    line1, = ax.plot(ga_fitness_history, color=colors['medium'],
                    linewidth=2.5, label=f'AG (Valor: {ga_value:.4f})' if ga_constant 
                    else 'Algoritmo Genético')
    line2, = ax.plot(hs_fitness_history, color=colors['dark'],
                    linewidth=2.5, label=f'HS (Valor: {hs_value:.4f})' if hs_constant 
                    else 'Harmony Search')
    
    if ga_constant and hs_constant:
        mean_value = (ga_value + hs_value) / 2
        ax.set_ylim(mean_value - 0.1, mean_value + 0.1)
        
        ax.text(0.02, 0.98, 'Nota: Ambos algoritmos alcanzaron y mantuvieron\n'
                'un valor constante desde el inicio.',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        all_fitness = ga_fitness_history + hs_fitness_history
        min_fitness = min(all_fitness)
        max_fitness = max(all_fitness)
        margin = (max_fitness - min_fitness) * 0.1 if max_fitness != min_fitness else 0.1
        ax.set_ylim(max(0, min_fitness - margin), max_fitness + margin)
    
    ax.set_title('Evolución del Fitness',
                pad=20, fontweight='bold', color=colors['text'], fontsize=16)
    ax.set_xlabel('Iteraciones', color=colors['text'], fontsize=14)
    ax.set_ylabel('Fitness', color=colors['text'], fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return [line1, line2]

def plot_execution_time(ga_time: float, hs_time: float, results_dir: str, 
                       base_filename: str, colors: Dict[str, str], logo_path: str):
    """
    Genera la gráfica comparativa de tiempos de ejecución.
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        generate_metric_plot(ax, ga_time, hs_time, 
                           'Comparación de Tiempos de Ejecución', 
                           'Tiempo (segundos)', colors)
        
        legend_elements = [
            patches.Patch(facecolor=colors['medium'], label='Algoritmo Genético'),
            patches.Patch(facecolor=colors['dark'], label='Harmony Search')
        ]
        add_logo_and_legend(fig, ax, logo_path, legend_elements)
        
        plt.savefig(os.path.join(results_dir, f'comparacion_tiempo_{base_filename}.png'),
                   dpi=300, bbox_inches='tight', facecolor=colors['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error al generar gráfica de tiempo: {str(e)}")
        plt.close()

def plot_generations(ga_gens: int, hs_gens: int, results_dir: str, 
                    base_filename: str, colors: Dict[str, str], logo_path: str):
    """
    Genera la gráfica comparativa del número de generaciones.
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        generate_metric_plot(ax, ga_gens, hs_gens, 
                           'Comparación de Número de Generaciones', 
                           'Generaciones', colors)
        
        legend_elements = [
            patches.Patch(facecolor=colors['medium'], label='Algoritmo Genético'),
            patches.Patch(facecolor=colors['dark'], label='Harmony Search')
        ]
        add_logo_and_legend(fig, ax, logo_path, legend_elements)
        
        plt.savefig(os.path.join(results_dir, f'comparacion_generaciones_{base_filename}.png'),
                   dpi=300, bbox_inches='tight', facecolor=colors['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error al generar gráfica de generaciones: {str(e)}")
        plt.close()

def plot_fitness_evolution(ga_fitness_history: List[float], hs_fitness_history: List[float],
                         results_dir: str, base_filename: str, colors: Dict[str, str],
                         logo_path: str):
    """
    Genera la gráfica de evolución del fitness.
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        legend_elements = generate_fitness_evolution_plot(
            ga_fitness_history, hs_fitness_history, ax, colors)
        
        add_logo_and_legend(fig, ax, logo_path, legend_elements)
        
        plt.savefig(os.path.join(results_dir, f'evolucion_fitness_{base_filename}.png'),
                   dpi=300, bbox_inches='tight', facecolor=colors['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error al generar gráfica de evolución del fitness: {str(e)}")
        plt.close()

def generate_professional_plots(ga_results: tuple, hs_results: tuple,
                              results_dir: str, base_filename: str,
                              logo_path: str = 'logoui1.png'):
    """
    Genera el conjunto principal de gráficas comparativas.
    
    Args:
        ga_results (tuple): Resultados del algoritmo genético
        hs_results (tuple): Resultados del harmony search
        results_dir (str): Directorio donde guardar las gráficas
        base_filename (str): Nombre base para los archivos
        logo_path (str): Ruta al archivo del logo
    """
    try:
        ga_fitness_history, ga_time, ga_gens, _ = ga_results
        hs_fitness_history, hs_time, hs_gens, _ = hs_results
        colors = get_red_palette()
        set_custom_style()

        # Generar cada tipo de gráfica
        plot_execution_time(ga_time, hs_time, results_dir, base_filename, colors, logo_path)
        plot_generations(ga_gens, hs_gens, results_dir, base_filename, colors, logo_path)
        plot_fitness_evolution(ga_fitness_history, hs_fitness_history, results_dir, base_filename, colors, logo_path)
        
        print("Gráficas profesionales generadas exitosamente")
        
    except Exception as e:
        print(f"Error general al generar gráficas: {str(e)}")
    finally:
        plt.close('all')

def generate_additional_plots(ga_results: tuple, hs_results: tuple,
                            results_dir: str, base_filename: str,
                            logo_path: str = 'logoui1.png'):
    """
    Genera gráficas adicionales para análisis detallado.
    
    Args:
        ga_results (tuple): Resultados del algoritmo genético
        hs_results (tuple): Resultados del harmony search
        results_dir (str): Directorio donde guardar las gráficas
        base_filename (str): Nombre base para los archivos
        logo_path (str): Ruta al archivo del logo
    """
    try:
        ga_fitness_history, _, _, _ = ga_results
        hs_fitness_history, _, _, _ = hs_results
        colors = get_red_palette()
        set_custom_style()

        # Calcular mejoras
        ga_improvements = calculate_improvements(ga_fitness_history)
        hs_improvements = calculate_improvements(hs_fitness_history)
        
        # Generar gráficas adicionales usando los helpers
        from visualization_helpers import (
            plot_improvement_rate,
            plot_improvement_distribution,
            plot_convergence_analysis
        )
        
        plot_improvement_rate(ga_improvements, hs_improvements, results_dir, base_filename, colors, logo_path)
        plot_improvement_distribution(ga_improvements, hs_improvements, results_dir, base_filename, colors, logo_path)
        plot_convergence_analysis(ga_fitness_history, hs_fitness_history, results_dir, base_filename, colors, logo_path)
        
        print("Gráficas adicionales generadas exitosamente")
        
    except Exception as e:
        print(f"Error general al generar gráficas adicionales: {str(e)}")
    finally:
        plt.close('all')        

# visualization.py - Corrección de funciones principales

def generate_professional_plots(ga_results: tuple, hs_results: tuple,
                              results_dir: str, base_filename: str,
                              logo_path: str = 'logoui1.png'):
    """
    Genera el conjunto principal de gráficas comparativas.
    
    Args:
        ga_results (tuple): Resultados del AG (best_solution, fitness_history, time, gens, start_time)
        hs_results (tuple): Resultados del HS (best_solution, fitness_history, time, gens, start_time)
        results_dir (str): Directorio donde guardar las gráficas
        base_filename (str): Nombre base para los archivos
        logo_path (str): Ruta al archivo del logo
    """
    try:
        # Desempaquetar correctamente los 5 valores
        ga_solution, ga_fitness_history, ga_time, ga_gens, ga_start_time = ga_results
        hs_solution, hs_fitness_history, hs_time, hs_gens, hs_start_time = hs_results
        
        colors = get_red_palette()
        set_custom_style()

        # Generar cada tipo de gráfica
        plot_execution_time(ga_time, hs_time, results_dir, base_filename, colors, logo_path)
        plot_generations(ga_gens, hs_gens, results_dir, base_filename, colors, logo_path)
        plot_fitness_evolution(ga_fitness_history, hs_fitness_history, results_dir, base_filename, colors, logo_path)
        
        print("Gráficas profesionales generadas exitosamente")
        
    except Exception as e:
        print(f"Error general al generar gráficas: {str(e)}")
    finally:
        plt.close('all')

def generate_additional_plots(ga_results: tuple, hs_results: tuple,
                            results_dir: str, base_filename: str,
                            logo_path: str = 'logoui1.png'):
    """
    Genera gráficas adicionales para análisis detallado.
    
    Args:
        ga_results (tuple): Resultados del AG (best_solution, fitness_history, time, gens, start_time)
        hs_results (tuple): Resultados del HS (best_solution, fitness_history, time, gens, start_time)
        results_dir (str): Directorio donde guardar las gráficas
        base_filename (str): Nombre base para los archivos
        logo_path (str): Ruta al archivo del logo
    """
    try:
        # Desempaquetar correctamente los 5 valores
        ga_solution, ga_fitness_history, ga_time, ga_gens, ga_start_time = ga_results
        hs_solution, hs_fitness_history, hs_time, hs_gens, hs_start_time = hs_results
        
        colors = get_red_palette()
        set_custom_style()

        # Calcular mejoras
        ga_improvements = calculate_improvements(ga_fitness_history)
        hs_improvements = calculate_improvements(hs_fitness_history)
        
        # Generar gráficas adicionales usando los helpers
        from visualization_helpers import (
            plot_improvement_rate,
            plot_improvement_distribution,
            plot_convergence_analysis
        )
        
        plot_improvement_rate(ga_improvements, hs_improvements, results_dir, base_filename, colors, logo_path)
        plot_improvement_distribution(ga_improvements, hs_improvements, results_dir, base_filename, colors, logo_path)
        plot_convergence_analysis(ga_fitness_history, hs_fitness_history, results_dir, base_filename, colors, logo_path)
        
        print("Gráficas adicionales generadas exitosamente")
        
    except Exception as e:
        print(f"Error general al generar gráficas adicionales: {str(e)}")
    finally:
        plt.close('all')        