"""
Módulo auxiliar con funciones helper para la generación de gráficas adicionales
del análisis comparativo de algoritmos de optimización.

Este módulo complementa al módulo principal de visualización, proporcionando
funciones específicas para gráficas más detalladas y complejas.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict

def plot_improvement_rate(ga_improvements: List[float],
                        hs_improvements: List[float],
                        results_dir: str,
                        base_filename: str,
                        colors: Dict[str, str],
                        logo_path: str):
    """
    Genera la gráfica de tasa de mejora por iteración.
    
    Args:
        ga_improvements: Lista de mejoras entre iteraciones del AG
        hs_improvements: Lista de mejoras entre iteraciones del HS
        results_dir: Directorio donde guardar la gráfica
        base_filename: Nombre base del archivo
        colors: Diccionario de colores
        logo_path: Ruta al archivo del logo
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        # Crear líneas de mejora
        line1, = ax.plot(ga_improvements, color=colors['medium'], 
                        linewidth=2.5, alpha=0.9, label='Algoritmo Genético')
        line2, = ax.plot(hs_improvements, color=colors['dark'], 
                        linewidth=2.5, alpha=0.9, label='Harmony Search')
        
        # Configurar gráfica
        ax.set_title('Tasa de Mejora por Iteración', 
                    pad=20, fontweight='bold', color=colors['text'], fontsize=16)
        ax.set_xlabel('Iteraciones', color=colors['text'], fontsize=14)
        ax.set_ylabel('Mejora en Fitness', color=colors['text'], fontsize=14)
        
        # Añadir leyenda y logo
        from visualization import add_logo_and_legend
        legend_elements = [line1, line2]
        add_logo_and_legend(fig, ax, logo_path, legend_elements)
        
        # Guardar gráfica
        plt.savefig(os.path.join(results_dir, f'tasa_mejora_{base_filename}.png'),
                    dpi=300, bbox_inches='tight', facecolor=colors['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error al generar gráfica de tasa de mejora: {str(e)}")
        plt.close()

def plot_improvement_distribution(ga_improvements: List[float],
                               hs_improvements: List[float],
                               results_dir: str,
                               base_filename: str,
                               colors: Dict[str, str],
                               logo_path: str):
    """
    Genera el histograma de distribución de mejoras.
    
    Args:
        ga_improvements: Lista de mejoras entre iteraciones del AG
        hs_improvements: Lista de mejoras entre iteraciones del HS
        results_dir: Directorio donde guardar la gráfica
        base_filename: Nombre base del archivo
        colors: Diccionario de colores
        logo_path: Ruta al archivo del logo
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        # Crear histogramas
        ax.hist(ga_improvements, bins=30, alpha=0.5, color=colors['medium'], 
                label='Algoritmo Genético')
        ax.hist(hs_improvements, bins=30, alpha=0.5, color=colors['dark'], 
                label='Harmony Search')
        
        # Configurar gráfica
        ax.set_title('Distribución de Mejoras', 
                    pad=20, fontweight='bold', color=colors['text'], fontsize=16)
        ax.set_xlabel('Magnitud de Mejora', color=colors['text'], fontsize=14)
        ax.set_ylabel('Frecuencia', color=colors['text'], fontsize=14)
        
        # Añadir leyenda y logo
        from visualization import add_logo_and_legend
        legend_elements = [
            patches.Patch(facecolor=colors['medium'], alpha=0.5, label='Algoritmo Genético'),
            patches.Patch(facecolor=colors['dark'], alpha=0.5, label='Harmony Search')
        ]
        add_logo_and_legend(fig, ax, logo_path, legend_elements)
        
        # Guardar gráfica
        plt.savefig(os.path.join(results_dir, f'distribucion_mejoras_{base_filename}.png'),
                    dpi=300, bbox_inches='tight', facecolor=colors['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error al generar histograma de mejoras: {str(e)}")
        plt.close()

def plot_convergence_analysis(ga_fitness_history: List[float],
                            hs_fitness_history: List[float],
                            results_dir: str,
                            base_filename: str,
                            colors: Dict[str, str],
                            logo_path: str):
    """
    Genera la gráfica de análisis de convergencia.
    
    Args:
        ga_fitness_history: Historial de fitness del AG
        hs_fitness_history: Historial de fitness del HS
        results_dir: Directorio donde guardar la gráfica
        base_filename: Nombre base del archivo
        colors: Diccionario de colores
        logo_path: Ruta al archivo del logo
    """
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        # Normalizar valores de fitness
        from visualization import safe_normalize
        ga_normalized = safe_normalize(ga_fitness_history)
        hs_normalized = safe_normalize(hs_fitness_history)
        
        # Crear líneas
        line1, = ax.plot(ga_normalized, color=colors['medium'], 
                        linewidth=2.5, alpha=0.9, label='Algoritmo Genético')
        line2, = ax.plot(hs_normalized, color=colors['dark'], 
                        linewidth=2.5, alpha=0.9, label='Harmony Search')
        
        # Configurar gráfica
        ax.set_title('Velocidad de Convergencia Normalizada', 
                    pad=20, fontweight='bold', color=colors['text'], fontsize=16)
        ax.set_xlabel('Iteraciones', color=colors['text'], fontsize=14)
        ax.set_ylabel('Progreso Normalizado', color=colors['text'], fontsize=14)
        
        # Añadir leyenda y logo
        from visualization import add_logo_and_legend
        legend_elements = [line1, line2]
        add_logo_and_legend(fig, ax, logo_path, legend_elements)
        
        # Guardar gráfica
        plt.savefig(os.path.join(results_dir, f'convergencia_{base_filename}.png'),
                    dpi=300, bbox_inches='tight', facecolor=colors['background'])
        plt.close()
        
    except Exception as e:
        print(f"Error al generar gráfica de convergencia: {str(e)}")
        plt.close()