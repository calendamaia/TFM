"""
Explorador 3D del Espacio de Búsqueda para Algoritmos Evolutivos.
Visualiza y manipula el espacio de soluciones en tiempo real.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
Fecha: 2024
"""
import os
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay
from typing import List, Dict, Tuple, Optional

# Colores corporativos UI1
COLORS = {
    'primary': '#E31837',       # Rojo UI1
    'primary_light': '#FF5F76', # Rojo claro
    'secondary': '#87CEEB',     # Azul para AG
    'background': '#FFFFFF',    # Blanco
    'text': '#4A4A4A',          # Gris oscuro
}

class SolutionSpace:
    """Clase para modelar y visualizar el espacio de soluciones."""
    
    def __init__(self):
        """Inicializa el modelador del espacio de soluciones."""
        # Configuración de directorios
        self.project_root = Path(__file__).parent
        # Modificar esta línea para apuntar a la carpeta correcta
        self.synthetic_data_dir = self.project_root / "datos_sinteticos" / "procesados"
        # Verificar existencia del directorio
        if not self.synthetic_data_dir.exists():
            os.makedirs(self.synthetic_data_dir, exist_ok=True)
            print(f"Creado directorio: {self.synthetic_data_dir}")
        # Rangos de parámetros del problema
        self.params = {
            'num_students': {'min': 5, 'max': 200, 'step': 1},
            'num_professors': {'min': 5, 'max': 60, 'step': 1},
            'num_buildings': {'min': 1, 'max': 3, 'step': 1},
            'num_classrooms': {'min': 1, 'max': 5, 'step': 1},
            'availability_rate': {'min': 0.3, 'max': 0.7, 'step': 0.01},
            'compatibility_rate': {'min': 0.3, 'max': 0.6, 'step': 0.01}
        }
        
        # Comportamiento modelado de los algoritmos
        self.behavior_models = {
            'AG': {
                'time_model': lambda complexity: 10 + 1.15**complexity * 20 + np.random.normal(0, 5),
                'generations_model': lambda complexity: 100 - 60 * np.exp(-0.2 * complexity) + np.random.normal(0, 5),
                'fitness_model': lambda complexity: 70 + 30 * (1 - np.exp(-0.2 * complexity)) + np.random.normal(0, 2)
            },
            'HS': {
                'time_model': lambda complexity: 5 + complexity * 3 + np.random.normal(0, 2),
                'generations_model': lambda complexity: 20 + np.random.normal(0, 2),
                'fitness_model': lambda complexity: 75 + 25 * (1 - np.exp(-0.15 * complexity)) + np.random.normal(0, 2)
            }
        }
        
        # Estructura para almacenar soluciones actuales
        self.current_solutions = {
            'AG': None,
            'HS': None
        }
        
        # Historial de exploración
        self.exploration_history = []
    
    def calculate_complexity(self, params: Dict) -> float:
        """
        Calcula el índice de complejidad basado en los parámetros.
        
        Args:
            params: Diccionario con los parámetros del problema
            
        Returns:
            Índice de complejidad normalizado
        """
        # Componentes de complejidad
        students = params.get('num_students', 10)
        professors = params.get('num_professors', 5)
        buildings = params.get('num_buildings', 1)
        classrooms = params.get('num_classrooms', 1)
        availability = params.get('availability_rate', 0.5)
        compatibility = params.get('compatibility_rate', 0.4)
        
        # Factores de complejidad
        assignment_complexity = students * professors
        spatial_complexity = buildings * classrooms * 10  # 10 slots por día en promedio
        constraint_complexity = (1 - availability) * (1 - compatibility) * 100
        
        # Índice compuesto normalizado (escala aproximada 0-10)
        return np.sqrt(assignment_complexity * spatial_complexity) / 100 + constraint_complexity
    
    def generate_solution(self, algorithm: str, params: Dict) -> Dict:
        """
        Genera una solución para los parámetros dados.
        
        Args:
            algorithm: Algoritmo a usar ('AG' o 'HS')
            params: Parámetros del problema
            
        Returns:
            Diccionario con la solución generada
        """
        # Calcular índice de complejidad
        complexity = self.calculate_complexity(params)
        
        # Obtener modelo de comportamiento
        model = self.behavior_models.get(algorithm)
        if not model:
            raise ValueError(f"Algoritmo desconocido: {algorithm}")
            
        # Generar métricas de la solución
        time = max(0.1, model['time_model'](complexity))
        generations = max(1, int(model['generations_model'](complexity)))
        fitness = max(0, min(100, model['fitness_model'](complexity)))
        
        # Calcular más métricas derivadas
        efficiency = fitness / time if time > 0 else 0
        convergence_rate = fitness / generations if generations > 0 else 0
        
        # Crear solución
        solution = {
            'algorithm': algorithm,
            'params': params.copy(),
            'complexity': complexity,
            'time': time,
            'generations': generations,
            'fitness': fitness,
            'efficiency': efficiency,
            'convergence_rate': convergence_rate
        }
        
        # Guardar en soluciones actuales
        self.current_solutions[algorithm] = solution
        
        # Añadir a historial
        self.exploration_history.append(solution)
        
        return solution
    
    def generate_space_samples(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Genera muestras del espacio de soluciones.
        
        Args:
            num_samples: Número de muestras a generar
            
        Returns:
            DataFrame con las muestras generadas
        """
        np.random.seed(42)  # Para reproducibilidad
        
        samples = []
        for _ in range(num_samples):
            # Generar parámetros aleatorios
            params = {
                'num_students': np.random.randint(
                    self.params['num_students']['min'],
                    self.params['num_students']['max']
                ),
                'num_professors': np.random.randint(
                    self.params['num_professors']['min'],
                    self.params['num_professors']['max']
                ),
                'num_buildings': np.random.randint(
                    self.params['num_buildings']['min'],
                    self.params['num_buildings']['max']
                ),
                'num_classrooms': np.random.randint(
                    self.params['num_classrooms']['min'],
                    self.params['num_classrooms']['max']
                ),
                'availability_rate': np.random.uniform(
                    self.params['availability_rate']['min'],
                    self.params['availability_rate']['max']
                ),
                'compatibility_rate': np.random.uniform(
                    self.params['compatibility_rate']['min'],
                    self.params['compatibility_rate']['max']
                )
            }
            
            # Ajustar número de profesores basado en estudiantes (mínimo 1/3)
            min_profs = max(5, params['num_students'] // 3)
            params['num_professors'] = max(min_profs, params['num_professors'])
            
            # Generar soluciones para ambos algoritmos
            ag_solution = self.generate_solution('AG', params)
            hs_solution = self.generate_solution('HS', params)
            
            # Añadir a muestras
            samples.append({
                'num_students': params['num_students'],
                'num_professors': params['num_professors'],
                'num_buildings': params['num_buildings'],
                'num_classrooms': params['num_classrooms'],
                'availability_rate': params['availability_rate'],
                'compatibility_rate': params['compatibility_rate'],
                'complexity': ag_solution['complexity'],  # Mismo para ambos
                'AG_time': ag_solution['time'],
                'HS_time': hs_solution['time'],
                'AG_generations': ag_solution['generations'],
                'HS_generations': hs_solution['generations'],
                'AG_fitness': ag_solution['fitness'],
                'HS_fitness': hs_solution['fitness'],
                'AG_efficiency': ag_solution['efficiency'],
                'HS_efficiency': hs_solution['efficiency']
            })
        
        return pd.DataFrame(samples)
    
    def create_3d_space_visualization(self, 
                                    df: pd.DataFrame,
                                    x_param: str,
                                    y_param: str,
                                    z_param: str,
                                    color_by: str = 'algorithm_advantage') -> go.Figure:
        """
        Crea una visualización 3D del espacio de soluciones.
        
        Args:
            df: DataFrame con las muestras
            x_param, y_param, z_param: Parámetros para los ejes
            color_by: Método de coloración ('algorithm_advantage' o 'complexity')
            
        Returns:
            Figura 3D de Plotly
        """
        # Crear figura base
        fig = go.Figure()
        
        # Determinar qué algoritmo tiene mejor rendimiento en cada punto
        if 'AG_' in z_param and 'HS_' in z_param:
            # Comparamos directamente dos algoritmos
            z_ag = df[z_param.replace('HS_', 'AG_')].values
            z_hs = df[z_param.replace('AG_', 'HS_')].values
            
            # Mejor algoritmo depende de la métrica
            if 'time' in z_param or 'generations' in z_param:
                # Menor es mejor
                df['better_algorithm'] = np.where(z_ag < z_hs, 'AG', 'HS')
                df['advantage_ratio'] = np.where(z_ag < z_hs, 
                                            z_hs / z_ag if z_ag > 0 else 1,
                                            z_ag / z_hs if z_hs > 0 else 1)
            else:
                # Mayor es mejor (fitness, efficiency)
                df['better_algorithm'] = np.where(z_ag > z_hs, 'AG', 'HS')
                df['advantage_ratio'] = np.where(z_ag > z_hs,
                                            z_ag / z_hs if z_hs > 0 else 1,
                                            z_hs / z_ag if z_ag > 0 else 1)
        else:
            # Estamos viendo un solo algoritmo o parámetro
            df['better_algorithm'] = 'Neutral'
            df['advantage_ratio'] = 1.0
        
        # Extraer valores para los ejes
        x = df[x_param].values
        y = df[y_param].values
        
        # Determinar valores para eje Z según parámetro
        if 'AG_' in z_param or 'HS_' in z_param:
            z = df[z_param].values
            z_label = z_param.split('_')[1].capitalize()
            if 'AG_' in z_param:
                algorithm_name = 'Algoritmo Genético'
                point_color = COLORS['secondary']
            else:
                algorithm_name = 'Harmony Search'
                point_color = COLORS['primary']
            z_title = f"{z_label} - {algorithm_name}"
        else:
            z = df[z_param].values
            z_title = z_param.replace('_', ' ').capitalize()
            point_color = '#8e44ad'  # Púrpura para parámetros neutrales
        
        # Colorear puntos según ventaja algorítmica o complejidad
        if color_by == 'algorithm_advantage' and 'better_algorithm' in df.columns:
            colors = np.where(df['better_algorithm'] == 'AG', 
                            COLORS['secondary'], 
                            np.where(df['better_algorithm'] == 'HS',
                                    COLORS['primary'],
                                    '#8e44ad'))  # Púrpura para neutral
            
            # Intensidad basada en la ventaja
            opacity = np.clip(df['advantage_ratio'] / 5, 0.3, 1.0)
            
        elif color_by == 'complexity':
            # Escala de color basada en complejidad
            normalized_complexity = (df['complexity'] - df['complexity'].min()) / \
                                (df['complexity'].max() - df['complexity'].min())
            colors = [f'rgb({int(255 * (1-c))}, {int(50 * (1-c))}, {int(200 * c)})' 
                    for c in normalized_complexity]
            opacity = 0.7
        else:
            # Color uniforme
            colors = point_color
            opacity = 0.7
        
        # Añadir puntos 3D
        hovertemplate = (
            f"{x_param}: %{{x}}<br>"
            f"{y_param}: %{{y}}<br>"
            f"{z_param}: %{{z:.2f}}<br>"
            "Complejidad: %{text}<br>"
            "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.7 if isinstance(opacity, pd.Series) else opacity,  # Valor fijo si es una serie
                    line=dict(width=1, color='#7F7F7F')
                ),
                hovertemplate=hovertemplate
            )
        )
        
        # Intentar crear superficies de tendencia si hay suficientes puntos
        if len(df) > 20 and 'AG_' in z_param:
            try:
                # Crear malla para superficie
                x_range = np.linspace(min(x), max(x), 20)
                y_range = np.linspace(min(y), max(y), 20)
                X, Y = np.meshgrid(x_range, y_range)
                
                # Interpolar valores Z para AG
                from scipy.interpolate import griddata
                Z_ag = griddata((x, y), df[z_param.replace('HS_', 'AG_')], (X, Y), method='cubic')
                
                # Añadir superficie AG
                fig.add_trace(
                    go.Surface(
                        x=X, y=Y, z=Z_ag,
                        colorscale='Blues',
                        opacity=0.6,
                        showscale=False,
                        name='Tendencia AG'
                    )
                )
                
                # Interpolar valores Z para HS
                Z_hs = griddata((x, y), df[z_param.replace('AG_', 'HS_')], (X, Y), method='cubic')
                
                # Añadir superficie HS
                fig.add_trace(
                    go.Surface(
                        x=X, y=Y, z=Z_hs,
                        colorscale='Reds',
                        opacity=0.6,
                        showscale=False,
                        name='Tendencia HS'
                    )
                )
                
                # Intentar encontrar la intersección de las superficies
                mask = np.abs(Z_ag - Z_hs) < 0.1 * np.mean(np.abs(Z_ag))
                if np.any(mask):
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
                            name='Frontera de rendimiento'
                        )
                    )
            except Exception as e:
                print(f"Error generando superficies: {str(e)}")
        
        # Configuración del layout 3D
        fig.update_layout(
            scene=dict(
                xaxis_title=x_param.replace('_', ' ').capitalize(),
                yaxis_title=y_param.replace('_', ' ').capitalize(),
                zaxis_title=z_title,
                xaxis=dict(showbackground=True, backgroundcolor='rgb(250, 250, 250)'),
                yaxis=dict(showbackground=True, backgroundcolor='rgb(250, 250, 250)'),
                zaxis=dict(showbackground=True, backgroundcolor='rgb(250, 250, 250)')
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        )
        
        return fig
    
    def find_optimal_parameters(self, 
                              df: pd.DataFrame, 
                              algorithm: str,
                              metric: str,
                              constraint_params: Optional[Dict] = None) -> Dict:
        """
        Encuentra los parámetros óptimos según alguna métrica.
        
        Args:
            df: DataFrame con muestras
            algorithm: Algoritmo a optimizar ('AG' o 'HS')
            metric: Métrica a optimizar ('time', 'generations', 'fitness', 'efficiency')
            constraint_params: Restricciones adicionales
            
        Returns:
            Diccionario con los parámetros óptimos
        """
        # Filtrar por restricciones si existen
        filtered_df = df.copy()
        if constraint_params:
            for param, value_range in constraint_params.items():
                if param in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[param] >= value_range[0]) & 
                        (filtered_df[param] <= value_range[1])
                    ]
        
        # Si no quedan filas después del filtrado, retornar None
        if filtered_df.empty:
            return None
        
        # Columna a optimizar
        target_col = f"{algorithm}_{metric}"
        
        # Determinar si mejor es mayor o menor valor
        if metric in ['time', 'generations']:
            # Menor es mejor
            optimal_idx = filtered_df[target_col].idxmin()
        else:
            # Mayor es mejor (fitness, efficiency)
            optimal_idx = filtered_df[target_col].idxmax()
        
        # Obtener fila óptima
        optimal_row = filtered_df.loc[optimal_idx]
        
        # Extraer parámetros óptimos
        optimal_params = {
            'num_students': int(optimal_row['num_students']),
            'num_professors': int(optimal_row['num_professors']),
            'num_buildings': int(optimal_row['num_buildings']),
            'num_classrooms': int(optimal_row['num_classrooms']),
            'availability_rate': float(optimal_row['availability_rate']),
            'compatibility_rate': float(optimal_row['compatibility_rate']),
            'complexity': float(optimal_row['complexity']),
            'metric_value': float(optimal_row[target_col])
        }
        
        return optimal_params

# Inicializar la aplicación Dash
app = dash.Dash(__name__, 
                title='Explorador 3D del Espacio de Búsqueda',
                external_stylesheets=[
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'
                ])
server = app.server

# Inicializar el modelador de espacio
solution_space = SolutionSpace()

# Generar datos para visualización inicial
space_samples = solution_space.generate_space_samples(num_samples=200)

# Definir layout
app.layout = html.Div([
    # Encabezado
    html.Div([
        html.Img(src='/assets/logoui1.png', 
                 style={'height': '60px', 'margin-right': '20px'}),
        html.H1("Explorador 3D del Espacio de Búsqueda",
                style={'color': COLORS['text'], 'flex-grow': '1'}),
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'padding': '15px 20px',
        'borderBottom': f'3px solid {COLORS["primary"]}',
        'backgroundColor': '#f8f8f8'
    }),
    
    # Panel principal
    html.Div([
        # Panel de controles
        html.Div([
            html.H3("Controles de exploración", 
                   style={'marginBottom': '20px', 'color': COLORS['text']}),
            
            # Parámetros de visualización
            html.Div([
                html.Label("Dimensiones:", style={'fontWeight': 'bold'}),
                
                # Selector eje X
                html.Div([
                    html.Label("Eje X:", className="control-label"),
                    dcc.Dropdown(
                        id='x-param',
                        options=[
                            {'label': 'Estudiantes', 'value': 'num_students'},
                            {'label': 'Profesores', 'value': 'num_professors'},
                            {'label': 'Edificios', 'value': 'num_buildings'},
                            {'label': 'Aulas', 'value': 'num_classrooms'},
                            {'label': 'Disponibilidad', 'value': 'availability_rate'},
                            {'label': 'Compatibilidad', 'value': 'compatibility_rate'},
                            {'label': 'Complejidad', 'value': 'complexity'}
                        ],
                        value='num_students',
                    ),
                ], className="control-group"),
                
                # Selector eje Y
                html.Div([
                    html.Label("Eje Y:", className="control-label"),
                    dcc.Dropdown(
                        id='y-param',
                        options=[
                            {'label': 'Estudiantes', 'value': 'num_students'},
                            {'label': 'Profesores', 'value': 'num_professors'},
                            {'label': 'Edificios', 'value': 'num_buildings'},
                            {'label': 'Aulas', 'value': 'num_classrooms'},
                            {'label': 'Disponibilidad', 'value': 'availability_rate'},
                            {'label': 'Compatibilidad', 'value': 'compatibility_rate'},
                            {'label': 'Complejidad', 'value': 'complexity'}
                        ],
                        value='num_professors',
                    ),
                ], className="control-group"),
                
                # Selector eje Z
                html.Div([
                    html.Label("Eje Z (Métrica):", className="control-label"),
                    dcc.Dropdown(
                        id='z-param',
                        options=[
                            {'label': 'AG - Tiempo', 'value': 'AG_time'},
                            {'label': 'HS - Tiempo', 'value': 'HS_time'},
                            {'label': 'AG - Generaciones', 'value': 'AG_generations'},
                            {'label': 'HS - Generaciones', 'value': 'HS_generations'},
                            {'label': 'AG - Fitness', 'value': 'AG_fitness'},
                            {'label': 'HS - Fitness', 'value': 'HS_fitness'},
                            {'label': 'AG - Eficiencia', 'value': 'AG_efficiency'},
                            {'label': 'HS - Eficiencia', 'value': 'HS_efficiency'}
                        ],
                        value='AG_time',
                    ),
                ], className="control-group"),
                
                # Selector de coloración
                html.Div([
                    html.Label("Colorear por:", className="control-label"),
                    dcc.RadioItems(
                        id='color-method',
                        options=[
                            {'label': 'Ventaja algorítmica', 'value': 'algorithm_advantage'},
                            {'label': 'Complejidad', 'value': 'complexity'}
                        ],
                        value='algorithm_advantage',
                        labelStyle={'display': 'block', 'margin': '5px 0'}
                    ),
                ], className="control-group"),
                
            ], style={'marginBottom': '30px'}),
            
            # Parámetros del problema para simulación
            html.Div([
                html.Label("Parámetros del problema:", style={'fontWeight': 'bold'}),
                
                # Deslizadores para cada parámetro
                html.Div([
                    html.Label("Estudiantes:", className="control-label"),
                    dcc.Slider(
                        id='students-slider',
                        min=solution_space.params['num_students']['min'],
                        max=solution_space.params['num_students']['max'],
                        step=solution_space.params['num_students']['step'],
                        value=50,
                        marks={i: str(i) for i in range(
                            solution_space.params['num_students']['min'],
                            solution_space.params['num_students']['max']+1,
                            50
                        )},
                    ),
                ], className="control-group slider-container"),
                
                html.Div([
                    html.Label("Profesores:", className="control-label"),
                    dcc.Slider(
                        id='professors-slider',
                        min=solution_space.params['num_professors']['min'],
                        max=solution_space.params['num_professors']['max'],
                        step=solution_space.params['num_professors']['step'],
                        value=20,
                        marks={i: str(i) for i in range(
                            solution_space.params['num_professors']['min'],
                            solution_space.params['num_professors']['max']+1,
                            10
                        )},
                    ),
                ], className="control-group slider-container"),
                
                html.Div([
                    html.Label("Edificios:", className="control-label"),
                    dcc.Slider(
                        id='buildings-slider',
                        min=solution_space.params['num_buildings']['min'],
                        max=solution_space.params['num_buildings']['max'],
                        step=solution_space.params['num_buildings']['step'],
                        value=1,
                        marks={i: str(i) for i in range(
                            solution_space.params['num_buildings']['min'],
                            solution_space.params['num_buildings']['max']+1,
                            1
                        )},
                    ),
                ], className="control-group slider-container"),
                
                html.Div([
                    html.Label("Aulas:", className="control-label"),
                    dcc.Slider(
                        id='classrooms-slider',
                        min=solution_space.params['num_classrooms']['min'],
                        max=solution_space.params['num_classrooms']['max'],
                        step=solution_space.params['num_classrooms']['step'],
                        value=2,
                        marks={i: str(i) for i in range(
                            solution_space.params['num_classrooms']['min'],
                            solution_space.params['num_classrooms']['max']+1,
                            1
                        )},
                    ),
                ], className="control-group slider-container"),
                
                html.Div([
                    html.Label("Disponibilidad:", className="control-label"),
                    dcc.Slider(
                        id='availability-slider',
                        min=solution_space.params['availability_rate']['min'],
                        max=solution_space.params['availability_rate']['max'],
                        step=solution_space.params['availability_rate']['step'],
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(
                            int(solution_space.params['availability_rate']['min']*10),
                            int(solution_space.params['availability_rate']['max']*10)+1,
                            1
                        )},
                    ),
                ], className="control-group slider-container"),
                
                html.Div([
                    html.Label("Compatibilidad:", className="control-label"),
                    dcc.Slider(
                        id='compatibility-slider',
                        min=solution_space.params['compatibility_rate']['min'],
                        max=solution_space.params['compatibility_rate']['max'],
                        step=solution_space.params['compatibility_rate']['step'],
                        value=0.4,
                        marks={i/10: str(i/10) for i in range(
                            int(solution_space.params['compatibility_rate']['min']*10),
                            int(solution_space.params['compatibility_rate']['max']*10)+1,
                            1
                        )},
                    ),
                ], className="control-group slider-container"),
                
            ], style={'marginBottom': '30px'}),
            
            # Botones de acción
            html.Div([
                html.Button(
                    "Actualizar visualización", 
                    id='update-viz-btn',
                    className="action-button primary"
                ),
                html.Button(
                    "Simular punto", 
                    id='simulate-point-btn',
                    className="action-button secondary"
                ),
                html.Button(
                    "Encontrar óptimo", 
                    id='find-optimal-btn',
                    className="action-button accent"
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),
            
            # Panel de resultados
            html.Div([
                html.H4("Resultados de simulación", 
                       style={'marginTop': '30px', 'marginBottom': '15px', 'color': COLORS['text']}),
                html.Div(id='simulation-results')
            ])
            
        ], style={
            'width': '300px',
            'padding': '20px',
            'backgroundColor': '#f0f0f0',
            'borderRight': '1px solid #ddd',
            'height': 'calc(100vh - 100px)',
            'overflowY': 'auto'
        }),
        
        # Panel de visualización
        html.Div([
            dcc.Loading(
                id="loading-visualization",
                type="circle",
                children=[
                    html.Div(id='visualization-container', style={
                        'height': 'calc(100vh - 100px)',
                        'display': 'flex',
                        'flexDirection': 'column'
                    })
                ]
            )
        ], style={
            'flex': '1',
            'height': 'calc(100vh - 100px)',
            'position': 'relative'
        })
    ], style={'display': 'flex', 'height': 'calc(100vh - 90px)'}),
    
    # Store para datos
    dcc.Store(id='space-data-store')
], style={'fontFamily': 'Arial', 'margin': 0, 'padding': 0})

# Cargar datos iniciales
@callback(
    Output('space-data-store', 'data'),
    Input('update-viz-btn', 'n_clicks'),
    prevent_initial_call=False
)
def load_initial_data(n_clicks):
    """Carga los datos iniciales o actualiza con nuevas muestras."""
    try:
        if n_clicks is None:
            return space_samples.to_json(date_format='iso', orient='split')
        else:
            # Generar nuevas muestras
            new_samples = solution_space.generate_space_samples(num_samples=200)
            return new_samples.to_json(date_format='iso', orient='split')
    except Exception as e:
        print(f"Error cargando datos: {str(e)}")
        # Retornar un conjunto de datos vacío pero válido
        return pd.DataFrame().to_json(date_format='iso', orient='split')

@callback(
    Output('visualization-container', 'children'),
    [Input('space-data-store', 'data'),
     Input('x-param', 'value'),
     Input('y-param', 'value'),
     Input('z-param', 'value'),
     Input('color-method', 'value')]
)
def update_visualization(data_json, x_param, y_param, z_param, color_method):
    """Actualiza la visualización 3D según los parámetros seleccionados."""
    if not data_json:
        return [html.Div("Cargando datos...")]
    
    # Convertir datos JSON a DataFrame
    df = pd.read_json(data_json, orient='split')
    
    # Generar visualización 3D
    fig = solution_space.create_3d_space_visualization(
        df, x_param, y_param, z_param, color_method
    )
    
    return [
        dcc.Graph(
            id='3d-space-plot',
            figure=fig,
            style={'height': 'calc(100vh - 100px)'}
        ),
        html.Div([
            html.Div([
                html.Span(className="legend-item ag"),
                html.Span("Algoritmo Genético dominante")
            ], className="legend-entry"),
            html.Div([
                html.Span(className="legend-item hs"),
                html.Span("Harmony Search dominante")
            ], className="legend-entry"),
            html.Div([
                html.Span(className="legend-item equal"),
                html.Span("Rendimiento similar")
            ], className="legend-entry")
        ], className="plot-legend")
    ]

@callback(
    Output('simulation-results', 'children'),
    [Input('simulate-point-btn', 'n_clicks')],
    [State('students-slider', 'value'),
     State('professors-slider', 'value'),
     State('buildings-slider', 'value'),
     State('classrooms-slider', 'value'),
     State('availability-slider', 'value'),
     State('compatibility-slider', 'value')]
)
def simulate_point(n_clicks, students, professors, buildings, classrooms, availability, compatibility):
    """Simula el rendimiento en un punto específico."""
    if n_clicks is None:
        return []
    
    # Crear parámetros
    params = {
        'num_students': students,
        'num_professors': professors,
        'num_buildings': buildings,
        'num_classrooms': classrooms,
        'availability_rate': availability,
        'compatibility_rate': compatibility
    }
    
    # Generar soluciones
    ag_solution = solution_space.generate_solution('AG', params)
    hs_solution = solution_space.generate_solution('HS', params)
    
    # Determinar ventajas
    time_advantage = "AG" if ag_solution['time'] < hs_solution['time'] else "HS"
    time_ratio = hs_solution['time'] / ag_solution['time'] if ag_solution['time'] > 0 else 0
    if time_advantage == "AG":
        time_text = f"AG es {time_ratio:.2f}x más rápido"
    else:
        time_text = f"HS es {1/time_ratio:.2f}x más rápido"
    
    gen_advantage = "AG" if ag_solution['generations'] < hs_solution['generations'] else "HS"
    gen_ratio = hs_solution['generations'] / ag_solution['generations'] if ag_solution['generations'] > 0 else 0
    if gen_advantage == "AG":
        gen_text = f"AG usa {gen_ratio:.2f}x menos generaciones"
    else:
        gen_text = f"HS usa {1/gen_ratio:.2f}x menos generaciones"
    
    fitness_advantage = "AG" if ag_solution['fitness'] > hs_solution['fitness'] else "HS"
    fitness_ratio = ag_solution['fitness'] / hs_solution['fitness'] if hs_solution['fitness'] > 0 else 0
    if fitness_advantage == "AG":
        fitness_text = f"AG logra {fitness_ratio:.2f}x mejor fitness"
    else:
        fitness_text = f"HS logra {1/fitness_ratio:.2f}x mejor fitness"
    
    # Crear resultados
    return [
        html.Div([
            html.H5(f"Complejidad: {ag_solution['complexity']:.2f}"),
            
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Métrica"),
                        html.Th("AG"),
                        html.Th("HS"),
                        html.Th("Ventaja")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td("Tiempo (s)"),
                        html.Td(f"{ag_solution['time']:.2f}"),
                        html.Td(f"{hs_solution['time']:.2f}"),
                        html.Td(time_text, style={'color': 
                               COLORS['secondary'] if time_advantage == 'AG' else COLORS['primary']})
                    ]),
                    html.Tr([
                        html.Td("Generaciones"),
                        html.Td(f"{ag_solution['generations']}"),
                        html.Td(f"{hs_solution['generations']}"),
                        html.Td(gen_text, style={'color': 
                               COLORS['secondary'] if gen_advantage == 'AG' else COLORS['primary']})
                    ]),
                    html.Tr([
                        html.Td("Fitness"),
                        html.Td(f"{ag_solution['fitness']:.2f}"),
                        html.Td(f"{hs_solution['fitness']:.2f}"),
                        html.Td(fitness_text, style={'color': 
                               COLORS['secondary'] if fitness_advantage == 'AG' else COLORS['primary']})
                    ]),
                    html.Tr([
                        html.Td("Eficiencia"),
                        html.Td(f"{ag_solution['efficiency']:.2f}"),
                        html.Td(f"{hs_solution['efficiency']:.2f}"),
                        html.Td("Fitness/Tiempo")
                    ])
                ])
            ], className="results-table"),
            
            html.Div([
                html.Strong("Conclusión: "),
                html.Span(
                    "Harmony Search" if hs_solution['efficiency'] > ag_solution['efficiency'] else "Algoritmo Genético",
                    style={'color': COLORS['primary'] if hs_solution['efficiency'] > ag_solution['efficiency'] 
                          else COLORS['secondary'], 'fontWeight': 'bold'}
                ),
                html.Span(" es más eficiente en este escenario.")
            ], className="conclusion")
        ])
    ]

@callback(
    Output('simulation-results', 'children', allow_duplicate=True),
    [Input('find-optimal-btn', 'n_clicks')],
    [State('space-data-store', 'data'),
     State('z-param', 'value')],
    prevent_initial_call=True
)
def find_optimal_parameters(n_clicks, data_json, z_param):
    """Encuentra los parámetros óptimos para la métrica seleccionada."""
    if n_clicks is None or not data_json:
        return []
    
    # Convertir datos JSON a DataFrame
    df = pd.read_json(data_json, orient='split')
    
    # Extraer algoritmo y métrica del z_param
    if '_' in z_param:
        algorithm, metric = z_param.split('_')
    else:
        # Si no hay algoritmo específico, usar ambos
        return find_optimal_comparison(df)
    
    # Encontrar parámetros óptimos
    optimal_params = solution_space.find_optimal_parameters(df, algorithm, metric)
    
    if not optimal_params:
        return [html.Div("No se encontraron parámetros óptimos con las restricciones actuales.")]
    
    # Crear resultados
    return [
        html.Div([
            html.H5(f"Parámetros óptimos para {algorithm} - {metric}"),
            
            html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td("Estudiantes:"),
                        html.Td(f"{optimal_params['num_students']}")
                    ]),
                    html.Tr([
                        html.Td("Profesores:"),
                        html.Td(f"{optimal_params['num_professors']}")
                    ]),
                    html.Tr([
                        html.Td("Edificios:"),
                        html.Td(f"{optimal_params['num_buildings']}")
                    ]),
                    html.Tr([
                        html.Td("Aulas:"),
                        html.Td(f"{optimal_params['num_classrooms']}")
                    ]),
                    html.Tr([
                        html.Td("Disponibilidad:"),
                        html.Td(f"{optimal_params['availability_rate']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("Compatibilidad:"),
                        html.Td(f"{optimal_params['compatibility_rate']:.2f}")
                    ]),
                    html.Tr([
                        html.Td("Complejidad:"),
                        html.Td(f"{optimal_params['complexity']:.2f}")
                    ]),
                    html.Tr([
                        html.Td(f"{metric.capitalize()}:"),
                        html.Td(f"{optimal_params['metric_value']:.2f}")
                    ])
                ])
            ], className="results-table"),
            
            html.Button(
                "Aplicar parámetros", 
                id='apply-optimal-btn',
                className="action-button secondary"
            )
        ])
    ]

def find_optimal_comparison(df):
    """Compara los mejores parámetros para ambos algoritmos."""
    metrics = ['time', 'generations', 'fitness', 'efficiency']
    results = {}
    
    for metric in metrics:
        for algorithm in ['AG', 'HS']:
            params = solution_space.find_optimal_parameters(df, algorithm, metric)
            if params:
                key = f"{algorithm}_{metric}"
                results[key] = params
    
    # Crear tabla comparativa
    return [
        html.Div([
            html.H5("Comparación de parámetros óptimos"),
            
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Métrica"),
                        html.Th("Algoritmo"),
                        html.Th("Valor"),
                        html.Th("Complejidad"),
                        html.Th("Estudiantes"),
                        html.Th("Profesores")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(metric.capitalize()),
                        html.Td(algorithm, style={'color': 
                              COLORS['secondary'] if algorithm == 'AG' else COLORS['primary']}),
                        html.Td(f"{results.get(f'{algorithm}_{metric}', {}).get('metric_value', '-'):.2f}"),
                        html.Td(f"{results.get(f'{algorithm}_{metric}', {}).get('complexity', '-'):.2f}"),
                        html.Td(f"{results.get(f'{algorithm}_{metric}', {}).get('num_students', '-')}"),
                        html.Td(f"{results.get(f'{algorithm}_{metric}', {}).get('num_professors', '-')}")
                    ])
                    for metric in metrics
                    for algorithm in ['AG', 'HS']
                    if f"{algorithm}_{metric}" in results
                ])
            ], className="results-table"),
            
            html.Div([
                html.Strong("Conclusión: "),
                html.Span(
                    "Harmony Search es óptimo para escenarios más complejos, mientras que el Algoritmo Genético destaca en problemas más simples.",
                    style={'fontStyle': 'italic'}
                )
            ], className="conclusion", style={'marginTop': '15px'})
        ])
    ]

# Estilos CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }
            .control-group {
                margin-bottom: 15px;
            }
            .control-label {
                display: block;
                margin-bottom: 5px;
                font-size: 14px;
            }
            .slider-container {
                padding: 0 10px;
            }
            .action-button {
                padding: 10px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            .action-button.primary {
                background-color: #E31837;
                color: white;
            }
            .action-button.secondary {
                background-color: #87CEEB;
                color: #4A4A4A;
            }
            .action-button.accent {
                background-color: #8e44ad;
                color: white;
            }
            .action-button:hover {
                opacity: 0.9;
            }
            .results-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                font-size: 14px;
            }
            .results-table th, .results-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .results-table th {
                background-color: #f5f5f5;
            }
            .results-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .conclusion {
                margin-top: 15px;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
            .plot-legend {
                position: absolute;
                bottom: 20px;
                right: 20px;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                z-index: 1000;
            }
            .legend-entry {
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }
            .legend-item {
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 10px;
                border-radius: 50%;
            }
            .legend-item.ag {
                background-color: #87CEEB;
            }
            .legend-item.hs {
                background-color: #E31837;
            }
            .legend-item.equal {
                background-color: #8e44ad;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)