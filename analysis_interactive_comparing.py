"""
Herramienta interactiva para comparación detallada entre algoritmos AG y HS.
Permite análisis paramétrico y visualización del espacio de búsqueda.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
Fecha: 2024
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import json
from scipy.spatial import ConvexHull

# Colores corporativos UI1
COLORS = {
    'primary': '#E31837',    # Rojo UI1 principal HS
    'secondary': '#87CEEB',  # Azul para AG
    'background': '#FFFFFF', # Blanco
    'text': '#4A4A4A',       # Gris oscuro
    'grid': '#F0F0F0',       # Gris claro para grilla
}

class SearchSpaceVisualizer:
    """Clase para visualización del espacio de búsqueda de los algoritmos."""
    
    def __init__(self):
        """Inicializa el visualizador del espacio de búsqueda."""
        self.dimensions = {
            'num_students': (5, 200),
            'num_professors': (5, 60),
            'num_buildings': (1, 3),
            'num_classrooms': (1, 5),
            'availability_rate': (0.3, 0.7),
            'compatibility_rate': (0.3, 0.6)
        }
        self.parameter_descriptions = {
            'num_students': 'Cantidad de estudiantes a programar',
            'num_professors': 'Cantidad de profesores disponibles',
            'num_buildings': 'Número de edificios para defensas',
            'num_classrooms': 'Aulas disponibles por edificio',
            'availability_rate': 'Tasa de disponibilidad temporal',
            'compatibility_rate': 'Tasa de compatibilidad estudiante-tribunal'
        }
        self.performance_metrics = {
            'tiempo': {'label': 'Tiempo de ejecución (s)', 'mejor': 'menor'},
            'generaciones': {'label': 'Generaciones necesarias', 'mejor': 'menor'},
            'fitness': {'label': 'Fitness alcanzado', 'mejor': 'mayor'},
            'eficiencia': {'label': 'Eficiencia (fitness/tiempo)', 'mejor': 'mayor'}
        }
        
    def generate_parameter_space(self, num_points=100):
        """
        Genera puntos representativos del espacio de parámetros.
        
        Args:
            num_points: Número de puntos a generar
            
        Returns:
            DataFrame con los puntos generados
        """
        np.random.seed(42)  # Para reproducibilidad
        
        points = {
            'num_students': np.random.randint(
                self.dimensions['num_students'][0],
                self.dimensions['num_students'][1],
                size=num_points
            ),
            'num_professors': [],
            'num_buildings': np.random.randint(
                self.dimensions['num_buildings'][0],
                self.dimensions['num_buildings'][1],
                size=num_points
            ),
            'num_classrooms': np.random.randint(
                self.dimensions['num_classrooms'][0],
                self.dimensions['num_classrooms'][1],
                size=num_points
            )
        }
        
        # Asegurar que profesores sea proporcional a estudiantes (1/3 como mínimo)
        for students in points['num_students']:
            min_profs = max(5, students // 3)
            max_profs = min(60, students // 2 + 10)
            if min_profs >= max_profs:
                max_profs = min_profs + 1  # Asegurar que max_profs sea mayor que min_profs
            points['num_professors'].append(np.random.randint(min_profs, max_profs + 1))
            
        # Tasas de disponibilidad y compatibilidad
        points['availability_rate'] = np.random.uniform(
            self.dimensions['availability_rate'][0],
            self.dimensions['availability_rate'][1],
            size=num_points
        )
        
        points['compatibility_rate'] = np.random.uniform(
            self.dimensions['compatibility_rate'][0],
            self.dimensions['compatibility_rate'][1],
            size=num_points
        )
        
        # Calcular índice de complejidad
        complexity = []
        for i in range(num_points):
            students = points['num_students'][i]
            professors = points['num_professors'][i]
            buildings = points['num_buildings'][i]
            classrooms = points['num_classrooms'][i]
            
            # Índice de complejidad compuesto
            complexity.append(np.sqrt((students * professors) * (buildings * classrooms)) / 100)
            
        # Simular tiempos y fitness según el índice de complejidad
        # Estas funciones simulan el comportamiento observado:
        # - AG tiempo crece exponencialmente con complejidad
        # - HS tiempo crece más linealmente
        # - Ambos fitness mejoran con complejidad, pero HS más consistentemente
        ag_time = []
        hs_time = []
        ag_generations = []
        hs_generations = []
        ag_fitness = []
        hs_fitness = []
        
        for c in complexity:
            # Comportamiento de tiempo
            if c <= 5:
                ag_t = 10 + (c/5) * 20 + np.random.normal(0, 5)
            else:
                ag_t = 30 + 1.07**(c-5) * 30 + np.random.normal(0, 30)
                
            hs_t = 5 + c * 8 + np.random.normal(0, 10)
            
            # Comportamiento de generaciones
            if c < 1:
                ag_g = np.random.randint(60, 90)
            elif c > 9:
                ag_g = np.random.randint(40, 60)
            else:
                ag_g = 100 + np.random.randint(-5, 6)
                
            hs_g = 20 + np.random.randint(-2, 3)
            
            # Comportamiento de fitness
            if c <= 5:
                ag_f = 70 + 30 * (1 - np.exp(-0.05 * c)) + np.random.normal(0, 3)
                hs_f = 65 + 25 * (1 - np.exp(-0.04 * c)) + np.random.normal(0, 3)
            else:
                ag_f = 90 + 10 * (1 - np.exp(-0.02 * (c-5))) + np.random.normal(0, 2)
                hs_f = 85 + 15 * (1 - np.exp(-0.03 * (c-5))) + np.random.normal(0, 2)
                
            # Limitar valores
            ag_time.append(max(5, min(550, ag_t)))
            hs_time.append(max(5, min(100, hs_t)))
            ag_generations.append(int(ag_g))
            hs_generations.append(int(hs_g))
            ag_fitness.append(max(50, min(110, ag_f)))
            hs_fitness.append(max(50, min(110, hs_f)))
            
        # Crear DataFrame
        df = pd.DataFrame({
            'num_students': points['num_students'],
            'num_professors': points['num_professors'],
            'num_buildings': points['num_buildings'],
            'num_classrooms': points['num_classrooms'],
            'availability_rate': points['availability_rate'],
            'compatibility_rate': points['compatibility_rate'],
            'complexity_index': complexity,
            'AG_time': ag_time,
            'HS_time': hs_time,
            'AG_generations': ag_generations,
            'HS_generations': hs_generations,
            'AG_fitness': ag_fitness,
            'HS_fitness': hs_fitness,
            'AG_efficiency': [f/t for f, t in zip(ag_fitness, ag_time)],
            'HS_efficiency': [f/t for f, t in zip(hs_fitness, hs_time)]
        })
        
        return df
    
    def get_winner_regions(self, df, metric='tiempo'):
        """
        Identifica regiones donde cada algoritmo es superior.
        
        Args:
            df: DataFrame con los datos de rendimiento
            metric: Métrica a analizar ('tiempo', 'generaciones', 'fitness', 'eficiencia')
            
        Returns:
            Tuple con puntos para las regiones donde gana AG y HS
        """
        # Determinar qué columnas comparar según la métrica
        if metric == 'tiempo':
            ag_col, hs_col = 'AG_time', 'HS_time'
            better = 'menor'  # Menor tiempo es mejor
        elif metric == 'generaciones':
            ag_col, hs_col = 'AG_generations', 'HS_generations'
            better = 'menor'  # Menos generaciones es mejor
        elif metric == 'fitness':
            ag_col, hs_col = 'AG_fitness', 'HS_fitness'
            better = 'mayor'  # Mayor fitness es mejor
        elif metric == 'eficiencia':
            ag_col, hs_col = 'AG_efficiency', 'HS_efficiency'
            better = 'mayor'  # Mayor eficiencia es mejor
        else:
            raise ValueError(f"Métrica desconocida: {metric}")
        
        # Determinar ganador en cada punto
        if better == 'menor':
            df['winner'] = np.where(df[ag_col] < df[hs_col], 'AG', 'HS')
        else:  # 'mayor'
            df['winner'] = np.where(df[ag_col] > df[hs_col], 'AG', 'HS')
        
        # Separar puntos donde gana cada algoritmo
        ag_points = df[df['winner'] == 'AG']
        hs_points = df[df['winner'] == 'HS']
        
        return ag_points, hs_points
    
    def create_convex_hull(self, points, parameters):
        """
        Crea una envolvente convexa para un conjunto de puntos.
        
        Args:
            points: DataFrame con los puntos
            parameters: Lista de dos parámetros para los ejes X e Y
            
        Returns:
            Array con los puntos de la envolvente convexa
        """
        if len(points) < 3:
            return None
            
        # Extraer puntos 2D para los parámetros seleccionados
        points_2d = points[[parameters[0], parameters[1]]].values
        
        try:
            # Calcular envolvente convexa
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            # Añadir el primer punto al final para cerrar el polígono
            hull_points = np.vstack([hull_points, hull_points[0]])
            return hull_points
        except Exception:
            # Si falla (e.g., puntos colineales), retornar None
            return None
    
    def generate_search_space_plot(self, df, params, metric='tiempo'):
        """
        Genera visualización del espacio de búsqueda para dos parámetros dados.
        
        Args:
            df: DataFrame con los datos
            params: Lista con dos nombres de parámetros para visualizar
            metric: Métrica a analizar
            
        Returns:
            Figura de Plotly
        """
        if len(params) != 2:
            raise ValueError("Se requieren exactamente dos parámetros")
            
        param_x, param_y = params
        
        # Obtener regiones donde gana cada algoritmo
        ag_points, hs_points = self.get_winner_regions(df, metric)
        
        # Crear figura base
        fig = go.Figure()
        
        # Determinar qué columnas comparar según la métrica
        if metric == 'tiempo':
            ag_col, hs_col = 'AG_time', 'HS_time'
            z_label = 'Tiempo de ejecución (s)'
        elif metric == 'generaciones':
            ag_col, hs_col = 'AG_generations', 'HS_generations'
            z_label = 'Número de generaciones'
        elif metric == 'fitness':
            ag_col, hs_col = 'AG_fitness', 'HS_fitness'
            z_label = 'Fitness alcanzado'
        elif metric == 'eficiencia':
            ag_col, hs_col = 'AG_efficiency', 'HS_efficiency'
            z_label = 'Eficiencia (fitness/tiempo)'
        
        # Calcular envolventes convexas
        ag_hull = self.create_convex_hull(ag_points, [param_x, param_y])
        hs_hull = self.create_convex_hull(hs_points, [param_x, param_y])
        
        # Añadir áreas de dominio si hay suficientes puntos
        if ag_hull is not None:
            fig.add_trace(
                go.Scatter(
                    x=ag_hull[:, 0],
                    y=ag_hull[:, 1],
                    fill="toself",
                    fillcolor=f"rgba(135, 206, 235, 0.3)",  # Azul semitransparente
                    line=dict(color=COLORS['secondary']),
                    name="Región óptima AG",
                    hoverinfo="skip"
                )
            )
            
        if hs_hull is not None:
            fig.add_trace(
                go.Scatter(
                    x=hs_hull[:, 0],
                    y=hs_hull[:, 1],
                    fill="toself",
                    fillcolor=f"rgba(227, 24, 55, 0.3)",  # Rojo semitransparente
                    line=dict(color=COLORS['primary']),
                    name="Región óptima HS",
                    hoverinfo="skip"
                )
            )
        
        # Añadir puntos para AG
        fig.add_trace(
            go.Scatter(
                x=ag_points[param_x],
                y=ag_points[param_y],
                mode='markers',
                marker=dict(
                    size=10,
                    color=ag_points[ag_col],
                    colorscale='Blues',
                    opacity=0.7,
                    colorbar=dict(
                        title=z_label,
                        x=0.45,
                        thickness=15
                    ),
                    showscale=True
                ),
                name='AG óptimo',
                hovertemplate=(
                    f"{param_x}: %{{x}}<br>"
                    f"{param_y}: %{{y}}<br>"
                    f"{z_label}: %{{marker.color:.2f}}"
                )
            )
        )
        
        # Añadir puntos para HS
        fig.add_trace(
            go.Scatter(
                x=hs_points[param_x],
                y=hs_points[param_y],
                mode='markers',
                marker=dict(
                    size=10,
                    color=hs_points[hs_col],
                    colorscale='Reds',
                    opacity=0.7,
                    colorbar=dict(
                        title=z_label,
                        x=1.0,
                        thickness=15
                    ),
                    showscale=True
                ),
                name='HS óptimo',
                hovertemplate=(
                    f"{param_x}: %{{x}}<br>"
                    f"{param_y}: %{{y}}<br>"
                    f"{z_label}: %{{marker.color:.2f}}"
                )
            )
        )
        
        # Configurar layout
        fig.update_layout(
            title=f"Espacio de búsqueda: Comparación AG vs HS ({self.performance_metrics[metric]['label']})",
            xaxis_title=f"{param_x} - {self.parameter_descriptions.get(param_x, '')}",
            yaxis_title=f"{param_y} - {self.parameter_descriptions.get(param_y, '')}",
            height=600,
            width=900,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',
            margin=dict(l=80, r=80, t=60, b=60),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white'
        )
        
        # Ajustar rangos
        if param_x in self.dimensions:
            fig.update_xaxes(range=[
                self.dimensions[param_x][0] * 0.9,
                self.dimensions[param_x][1] * 1.1
            ])
        
        if param_y in self.dimensions:
            fig.update_yaxes(range=[
                self.dimensions[param_y][0] * 0.9,
                self.dimensions[param_y][1] * 1.1
            ])
        
        return fig
    
    def generate_3d_performance_surface(self, df, params, metric='tiempo'):
        """
        Genera una superficie 3D mostrando el rendimiento de los algoritmos.
        
        Args:
            df: DataFrame con datos de rendimiento
            params: Lista con dos parámetros para los ejes X e Y
            metric: Métrica a visualizar
            
        Returns:
            Figura 3D de Plotly
        """
        if len(params) != 2:
            raise ValueError("Se requieren exactamente dos parámetros")
            
        param_x, param_y = params
        
        # Determinar qué columnas visualizar según la métrica
        if metric == 'tiempo':
            ag_col, hs_col = 'AG_time', 'HS_time'
            z_label = 'Tiempo de ejecución (s)'
        elif metric == 'generaciones':
            ag_col, hs_col = 'AG_generations', 'HS_generations'
            z_label = 'Número de generaciones'
        elif metric == 'fitness':
            ag_col, hs_col = 'AG_fitness', 'HS_fitness'
            z_label = 'Fitness alcanzado'
        elif metric == 'eficiencia':
            ag_col, hs_col = 'AG_efficiency', 'HS_efficiency'
            z_label = 'Eficiencia (fitness/tiempo)'
        else:
            raise ValueError(f"Métrica desconocida: {metric}")
        
        # Crear malla para interpolar superficies
        from scipy.interpolate import griddata
        
        x_vals = df[param_x].values
        y_vals = df[param_y].values
        
        # Crear malla regular
        x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_range = np.linspace(y_vals.min(), y_vals.max(), 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolar valores Z para ambos algoritmos
        Z_ag = griddata((x_vals, y_vals), df[ag_col].values, (X, Y), method='cubic')
        Z_hs = griddata((x_vals, y_vals), df[hs_col].values, (X, Y), method='cubic')
        
        # Crear figura 3D
        fig = go.Figure()
        
        # Superficie AG
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_ag,
                colorscale='Blues',
                name='AG',
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title=z_label,
                    x=0.45,
                    thickness=15
                ),
                hovertemplate=(
                    f"{param_x}: %{{x}}<br>"
                    f"{param_y}: %{{y}}<br>"
                    f"{z_label}: %{{z:.2f}}"
                )
            )
        )
        
        # Superficie HS
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_hs,
                colorscale='Reds',
                name='HS',
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title=z_label,
                    x=1.0,
                    thickness=15
                ),
                hovertemplate=(
                    f"{param_x}: %{{x}}<br>"
                    f"{param_y}: %{{y}}<br>"
                    f"{z_label}: %{{z:.2f}}"
                )
            )
        )
        
        # Puntos originales
        fig.add_trace(
            go.Scatter3d(
                x=x_vals, y=y_vals, z=df[ag_col].values,
                mode='markers',
                marker=dict(
                    size=4,
                    color=COLORS['secondary'],
                    opacity=0.5
                ),
                name='AG (puntos)',
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=x_vals, y=y_vals, z=df[hs_col].values,
                mode='markers',
                marker=dict(
                    size=4,
                    color=COLORS['primary'],
                    opacity=0.5
                ),
                name='HS (puntos)',
                hoverinfo='skip'
            )
        )
        
        # Plano de intersección (donde ambos algoritmos tienen el mismo rendimiento)
        # Crear puntos donde Z_ag == Z_hs (aprox)
        mask = np.abs(Z_ag - Z_hs) < (np.abs(Z_ag).mean() * 0.05)
        if mask.any():
            intersection_x = X[mask]
            intersection_y = Y[mask]
            intersection_z = (Z_ag[mask] + Z_hs[mask]) / 2
            
            fig.add_trace(
                go.Scatter3d(
                    x=intersection_x, y=intersection_y, z=intersection_z,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='green',
                        opacity=0.7
                    ),
                    name='PE',
                    hovertemplate=(
                        f"{param_x}: %{{x}}<br>"
                        f"{param_y}: %{{y}}<br>"
                        f"{z_label}: %{{z:.2f}}"
                    )
                )
            )
        
        # Configuración 3D
        fig.update_layout(
            title=f"Superficie de rendimiento 3D: {self.performance_metrics[metric]['label']}",
            scene=dict(
                xaxis_title=f"{param_x}",
                yaxis_title=f"{param_y}",
                zaxis_title=z_label,
                xaxis=dict(gridcolor='lightgray', showbackground=True, backgroundcolor='white'),
                yaxis=dict(gridcolor='lightgray', showbackground=True, backgroundcolor='white'),
                zaxis=dict(gridcolor='lightgray', showbackground=True, backgroundcolor='white')
            ),
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ddd',
                borderwidth=1
            )
        )
        
        return fig
    
    def generate_parameter_sensitivity_plot(self, df, param, metric='tiempo'):
        """
        Genera un gráfico de sensibilidad para un parámetro específico.
        
        Args:
            df: DataFrame con datos
            param: Parámetro a analizar
            metric: Métrica a visualizar
            
        Returns:
            Figura de Plotly
        """
        # Determinar columnas según métrica
        if metric == 'tiempo':
            ag_col, hs_col = 'AG_time', 'HS_time'
            y_label = 'Tiempo de ejecución (s)'
            mejor = 'menor'
        elif metric == 'generaciones':
            ag_col, hs_col = 'AG_generations', 'HS_generations'
            y_label = 'Número de generaciones'
            mejor = 'menor'
        elif metric == 'fitness':
            ag_col, hs_col = 'AG_fitness', 'HS_fitness'
            y_label = 'Fitness alcanzado'
            mejor = 'mayor'
        elif metric == 'eficiencia':
            ag_col, hs_col = 'AG_efficiency', 'HS_efficiency'
            y_label = 'Eficiencia (fitness/tiempo)'
            mejor = 'mayor'
        
        # Ordenar por el parámetro
        df_sorted = df.sort_values(param)
        
        # Crear figura
        fig = go.Figure()
        
        # Añadir líneas de tendencia
        fig.add_trace(
            go.Scatter(
                x=df_sorted[param],
                y=df_sorted[ag_col],
                mode='lines+markers',
                name='AG',
                line=dict(color=COLORS['secondary'], width=2),
                marker=dict(size=6, color=COLORS['secondary']),
                hovertemplate=f"{param}: %{{x}}<br>{y_label}: %{{y:.2f}}"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_sorted[param],
                y=df_sorted[hs_col],
                mode='lines+markers',
                name='HS',
                line=dict(color=COLORS['primary'], width=2),
                marker=dict(size=6, color=COLORS['primary']),
                hovertemplate=f"{param}: %{{x}}<br>{y_label}: %{{y:.2f}}"
            )
        )
        
        # Áreas de ventaja (destacar dónde cada algoritmo es mejor)
        if mejor == 'menor':
            advantage = df_sorted[ag_col] - df_sorted[hs_col]
        else:  # 'mayor'
            advantage = df_sorted[hs_col] - df_sorted[ag_col]
            
        crossover_points = []
        for i in range(1, len(advantage)):
            if advantage[i-1] * advantage[i] <= 0:  # Cambio de signo
                # Interpolar para encontrar el punto exacto
                x0, x1 = df_sorted[param].iloc[i-1], df_sorted[param].iloc[i]
                y0, y1 = advantage.iloc[i-1], advantage.iloc[i]
                
                if y1 - y0 != 0:
                    x_cross = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                    crossover_points.append(x_cross)
        
        # Añadir líneas verticales en los puntos de cruce
        for x_cross in crossover_points:
            fig.add_shape(
                type="line",
                x0=x_cross, y0=0,
                x1=x_cross, y1=df_sorted[[ag_col, hs_col]].max().max(),
                line=dict(color="green", width=1, dash="dash"),
                name="Punto de cruce"
            )
        
        # Configuración
        fig.update_layout(
            title=f"Análisis de sensibilidad: {param} vs {self.performance_metrics[metric]['label']}",
            xaxis_title=f"{param} - {self.parameter_descriptions.get(param, '')}",
            yaxis_title=y_label,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',
            margin=dict(l=80, r=50, t=60, b=60),
            font=dict(family="Arial", size=12),
            plot_bgcolor='white'
        )
        
        if len(crossover_points) > 0:
            annotations = []
            for x_cross in crossover_points:
                annotations.append(
                    dict(
                        x=x_cross,
                        y=df_sorted[[ag_col, hs_col]].max().max() * 0.5,
                        text="P", # Punto de equilibrio
                        showarrow=True,
                        arrowhead=2,
                        ax=40,
                        ay=0,
                        font=dict(color="green")
                    )
                )
            fig.update_layout(annotations=annotations)
        
        return fig

# Inicializar la aplicación Dash
app = dash.Dash(__name__, 
                title='Comparador Interactivo AG vs HS',
                external_stylesheets=[
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'
                ])
server = app.server

# Inicializar visualizador
visualizer = SearchSpaceVisualizer()

# Generar datos sintéticos
synthetic_data = visualizer.generate_parameter_space(num_points=150)

# Definir estructura del layout
app.layout = html.Div([
    # Encabezado
    html.Div([
        html.Img(src='/assets/logoui1.png', 
                 style={'height': '60px', 'margin-right': '20px'}),
        html.H1("Comparador Interactivo: Algoritmo Genético vs Harmony Search",
                style={'color': COLORS['text'], 'flex-grow': '1'}),
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'padding': '15px 20px',
        'borderBottom': f'3px solid {COLORS["primary"]}',
        'backgroundColor': '#f8f8f8'
    }),
    
    # Panel de controles y visualización
    html.Div([
        # Panel de controles
        html.Div([
            html.H3("Parámetros de análisis", style={'marginBottom': '20px'}),
            
            # Selección de visualización
            html.Div([
                html.Label("Tipo de visualización:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='visualization-type',
                    options=[
                        {'label': 'Espacio de búsqueda 2D', 'value': 'search-space-2d'},
                        {'label': 'Superficie de rendimiento 3D', 'value': 'performance-3d'},
                        {'label': 'Análisis de sensibilidad', 'value': 'sensitivity'}
                    ],
                    value='search-space-2d',
                    labelStyle={'display': 'block', 'margin': '5px 0'}
                ),
            ], style={'marginBottom': '25px'}),
            
            # Selección de parámetros - visible para espacio 2D y superficie 3D
            html.Div([
                html.Label("Eje X:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='param-x',
                    options=[
                        {'label': f"Estudiantes ({visualizer.parameter_descriptions['num_students']})", 
                         'value': 'num_students'},
                        {'label': f"Profesores ({visualizer.parameter_descriptions['num_professors']})", 
                         'value': 'num_professors'},
                        {'label': f"Edificios ({visualizer.parameter_descriptions['num_buildings']})", 
                         'value': 'num_buildings'},
                        {'label': f"Aulas ({visualizer.parameter_descriptions['num_classrooms']})", 
                         'value': 'num_classrooms'},
                        {'label': f"Tasa disponibilidad ({visualizer.parameter_descriptions['availability_rate']})", 
                         'value': 'availability_rate'},
                        {'label': f"Tasa compatibilidad ({visualizer.parameter_descriptions['compatibility_rate']})", 
                         'value': 'compatibility_rate'},
                        {'label': "Índice de complejidad", 'value': 'complexity_index'}
                    ],
                    value='num_students',
                ),
                
                html.Label("Eje Y:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                dcc.Dropdown(
                    id='param-y',
                    options=[
                        {'label': f"Estudiantes ({visualizer.parameter_descriptions['num_students']})", 
                         'value': 'num_students'},
                        {'label': f"Profesores ({visualizer.parameter_descriptions['num_professors']})", 
                         'value': 'num_professors'},
                        {'label': f"Edificios ({visualizer.parameter_descriptions['num_buildings']})", 
                         'value': 'num_buildings'},
                        {'label': f"Aulas ({visualizer.parameter_descriptions['num_classrooms']})", 
                         'value': 'num_classrooms'},
                        {'label': f"Tasa disponibilidad ({visualizer.parameter_descriptions['availability_rate']})", 
                         'value': 'availability_rate'},
                        {'label': f"Tasa compatibilidad ({visualizer.parameter_descriptions['compatibility_rate']})", 
                         'value': 'compatibility_rate'},
                        {'label': "Índice de complejidad", 'value': 'complexity_index'}
                    ],
                    value='num_professors',
                ),
            ], id='param-xy-container', style={'marginBottom': '25px'}),
            
            # Selección de parámetro único - visible para análisis de sensibilidad
            html.Div([
                html.Label("Parámetro a analizar:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='param-single',
                    options=[
                        {'label': f"Estudiantes ({visualizer.parameter_descriptions['num_students']})", 
                         'value': 'num_students'},
                        {'label': f"Profesores ({visualizer.parameter_descriptions['num_professors']})", 
                         'value': 'num_professors'},
                        {'label': f"Edificios ({visualizer.parameter_descriptions['num_buildings']})", 
                         'value': 'num_buildings'},
                        {'label': f"Aulas ({visualizer.parameter_descriptions['num_classrooms']})", 
                         'value': 'num_classrooms'},
                        {'label': f"Tasa disponibilidad ({visualizer.parameter_descriptions['availability_rate']})", 
                         'value': 'availability_rate'},
                        {'label': f"Tasa compatibilidad ({visualizer.parameter_descriptions['compatibility_rate']})", 
                         'value': 'compatibility_rate'},
                        {'label': "Índice de complejidad", 'value': 'complexity_index'}
                    ],
                    value='complexity_index',
                ),
            ], id='param-single-container', style={'display': 'none', 'marginBottom': '25px'}),
            
            # Selección de métrica
            html.Div([
                html.Label("Métrica de rendimiento:", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='metric-selector',
                    options=[
                        {'label': visualizer.performance_metrics['tiempo']['label'], 
                         'value': 'tiempo'},
                        {'label': visualizer.performance_metrics['generaciones']['label'], 
                         'value': 'generaciones'},
                        {'label': visualizer.performance_metrics['fitness']['label'], 
                         'value': 'fitness'},
                        {'label': visualizer.performance_metrics['eficiencia']['label'], 
                         'value': 'eficiencia'}
                    ],
                    value='tiempo',
                    labelStyle={'display': 'block', 'margin': '5px 0'}
                ),
            ], style={'marginBottom': '30px'}),
            
            # Número de puntos a mostrar (simulación)
            html.Div([
                html.Label("Densidad de puntos:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='num-points-slider',
                    min=10,
                    max=150,
                    step=10,
                    value=100,
                    marks={i: str(i) for i in range(10, 151, 20)},
                ),
            ], style={'marginBottom': '30px'}),
            
            # Botón de actualización
            html.Button(
                "Actualizar visualización", 
                id='update-viz-btn',
                style={
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 15px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%'
                }
            ),
            
            # Tabla de ventajas comparativas
            html.Div([
                html.H4("Ventaja comparativa", style={'marginTop': '30px', 'marginBottom': '10px'}),
                html.Div(id='advantage-table')
            ])
            
        ], style={
            'width': '300px',
            'padding': '20px',
            'backgroundColor': '#f0f0f0',
            'borderRight': '1px solid #ddd',
            'height': 'calc(100vh - 100px)',
            'overflowY': 'auto'
        }),
        
        # Área de visualización
        html.Div([
            dcc.Loading(
                id="loading-visualization",
                type="circle",
                children=[
                    html.Div(id='visualization-container')
                ]
            )
        ], style={
            'flex': '1',
            'padding': '20px',
            'height': 'calc(100vh - 100px)',
            'overflowY': 'auto',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
    ], style={'display': 'flex', 'height': 'calc(100vh - 90px)'})
])

@callback(
    [Output('param-xy-container', 'style'),
     Output('param-single-container', 'style')],
    [Input('visualization-type', 'value')]
)
def toggle_parameter_controls(viz_type):
    """Muestra u oculta controles de parámetros según el tipo de visualización."""
    if viz_type == 'sensitivity':
        return {'display': 'none'}, {'display': 'block', 'marginBottom': '25px'}
    else:
        return {'marginBottom': '25px'}, {'display': 'none'}

@callback(
    [Output('visualization-container', 'children'),
     Output('advantage-table', 'children')],
    [Input('update-viz-btn', 'n_clicks')],
    [State('visualization-type', 'value'),
     State('param-x', 'value'),
     State('param-y', 'value'),
     State('param-single', 'value'),
     State('metric-selector', 'value'),
     State('num-points-slider', 'value')]
)
def update_visualization(n_clicks, viz_type, param_x, param_y, param_single, metric, num_points):
    """Actualiza la visualización según los parámetros seleccionados."""
    if n_clicks is None:
        # Visualización inicial
        viz_type = 'search-space-2d'
        param_x = 'num_students'
        param_y = 'num_professors'
        metric = 'tiempo'
        num_points = 100
    
    # Generar datos según el número de puntos seleccionado
    df = visualizer.generate_parameter_space(num_points=num_points)
    
    # Generar tabla de ventajas
    advantage_table = generate_advantage_table(df, metric)
    
    # Crear visualización según el tipo seleccionado
    if viz_type == 'search-space-2d':
        fig = visualizer.generate_search_space_plot(df, [param_x, param_y], metric)
        return [
            dcc.Graph(
                id='main-visualization',
                figure=fig,
                style={'height': '700px', 'width': '900px'}
            )
        ], advantage_table
    
    elif viz_type == 'performance-3d':
        fig = visualizer.generate_3d_performance_surface(df, [param_x, param_y], metric)
        return [
            dcc.Graph(
                id='main-visualization',
                figure=fig,
                style={'height': '700px', 'width': '900px'}
            ),
            html.Div([
                html.H4("Instrucciones 3D:", 
                      style={'marginTop': '20px', 'color': COLORS['text']}),
                html.Ul([
                    html.Li("Arrastre para rotar la visualización"),
                    html.Li("Desplácese para acercar/alejar"),
                    html.Li("Doble clic para restablecer la vista")
                ], style={'fontSize': '14px'})
            ], style={'width': '900px', 'marginTop': '10px'})
        ], advantage_table
    
    elif viz_type == 'sensitivity':
        fig = visualizer.generate_parameter_sensitivity_plot(df, param_single, metric)
        return [
            dcc.Graph(
                id='main-visualization',
                figure=fig,
                style={'height': '500px', 'width': '900px'}
            ),
            html.Div([
                html.H4("Interpretación:", style={'marginTop': '20px', 'color': COLORS['text']}),
                html.P([
                    "Este gráfico muestra cómo cambia el rendimiento de cada algoritmo al variar el parámetro seleccionado. ",
                    html.Strong("Las líneas verticales verdes"), " indican los puntos donde ambos algoritmos tienen rendimiento similar. P=punto de equilibrio."
                ], style={'fontSize': '14px', 'lineHeight': '1.5', 'width': '900px'})
            ])
        ], advantage_table
    
    # Fallback
    return [html.Div("Tipo de visualización no implementado")], advantage_table

def generate_advantage_table(df, metric):
    """Genera una tabla con las ventajas comparativas entre algoritmos."""
    # Calcular métricas generales
    if metric == 'tiempo':
        ag_col, hs_col = 'AG_time', 'HS_time'
        mejor = 'menor'
    elif metric == 'generaciones': 
        ag_col, hs_col = 'AG_generations', 'HS_generations'
        mejor = 'menor'
    elif metric == 'fitness':
        ag_col, hs_col = 'AG_fitness', 'HS_fitness'
        mejor = 'mayor'
    elif metric == 'eficiencia':
        ag_col, hs_col = 'AG_efficiency', 'HS_efficiency'
        mejor = 'mayor'
    
    # Calcular promedios
    ag_avg = df[ag_col].mean()
    hs_avg = df[hs_col].mean()
    
    # Determinar ganador global
    if mejor == 'menor':
        global_winner = 'AG' if ag_avg < hs_avg else 'HS'
        improvement = abs(((hs_avg - ag_avg) / hs_avg) * 100) if hs_avg != 0 else 0
    else:  # 'mayor'
        global_winner = 'AG' if ag_avg > hs_avg else 'HS'
        improvement = abs(((ag_avg - hs_avg) / ag_avg) * 100) if ag_avg != 0 else 0
    
    # Contar victorias por complejidad
    df['winner'] = 'Empate'
    if mejor == 'menor':
        df.loc[df[ag_col] < df[hs_col], 'winner'] = 'AG'
        df.loc[df[hs_col] < df[ag_col], 'winner'] = 'HS'
    else:  # 'mayor'
        df.loc[df[ag_col] > df[hs_col], 'winner'] = 'AG'
        df.loc[df[hs_col] > df[ag_col], 'winner'] = 'HS'
    
    # Dividir por niveles de complejidad
    df['complexity_level'] = pd.cut(
        df['complexity_index'],
        bins=[0, 2, 5, 10, float('inf')],
        labels=['Baja', 'Media', 'Alta', 'Muy alta']
    )
    
    winners_by_complexity = df.groupby('complexity_level')['winner'].value_counts().unstack().fillna(0)
    #winners_by_complexity = df.groupby('complexity_level')['winner'].value_counts(observed=False).unstack().fillna(0)

    # Crear tabla HTML
    return html.Div([
        html.Div([
            html.Strong("Rendimiento promedio:"),
            html.Ul([
                html.Li(f"AG: {ag_avg:.2f}"),
                html.Li(f"HS: {hs_avg:.2f}")
            ], style={'paddingLeft': '20px', 'margin': '5px 0'})
        ]),
        
        html.Div([
            html.Strong(f"Ganador global: "),
            html.Span(
                global_winner,
                style={
                    'color': COLORS['secondary'] if global_winner == 'AG' else COLORS['primary'],
                    'fontWeight': 'bold'
                }
            ),
            html.Span(f" (mejora del {improvement:.1f}%)")
        ], style={'marginTop': '10px', 'marginBottom': '15px'}),
        
        html.Div([
            html.Strong("Rendimiento por complejidad:"),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Complejidad", style={'textAlign': 'left', 'padding': '5px'}),
                        html.Th("AG Gana", style={'textAlign': 'center', 'padding': '5px'}),
                        html.Th("HS Gana", style={'textAlign': 'center', 'padding': '5px'}),
                        html.Th("Empate", style={'textAlign': 'center', 'padding': '5px'})
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(complexity, style={'padding': '5px'}),
                        html.Td(
                            int(winners_by_complexity.get(('AG', complexity), 0)),
                            style={'textAlign': 'center', 'padding': '5px'}
                        ),
                        html.Td(
                            int(winners_by_complexity.get(('HS', complexity), 0)),
                            style={'textAlign': 'center', 'padding': '5px'}
                        ),
                        html.Td(
                            int(winners_by_complexity.get(('Empate', complexity), 0)),
                            style={'textAlign': 'center', 'padding': '5px'}
                        )
                    ]) for complexity in ['Baja', 'Media', 'Alta', 'Muy alta']
                ])
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'marginTop': '10px',
                'fontSize': '14px'
            })
        ])
    ], style={'marginTop': '15px'})

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)