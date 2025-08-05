"""
Módulo base para el problema de asignación de tribunales y horarios.

Define la estructura y funcionalidad común para los diferentes algoritmos
de optimización implementados.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from data_structures import TimeTableSolution

class TimetablingProblem:
    """
    Clase base que define la estructura y funcionalidad común para resolver
    el problema de asignación de tribunales y horarios.
    """
    
    def __init__(self, excel_path: str):
        """
        Inicializa el problema de timetabling cargando los datos necesarios.
        
        Args:
            excel_path (str): Ruta al archivo Excel con los datos de entrada
        """
        # Cargar datos del Excel
        self.excel_data = {
            'turnos': pd.read_excel(excel_path, sheet_name='Turnos'),
            'disp_alumnos_turnos': pd.read_excel(excel_path, sheet_name='Disponibilidad-alumnos-turnos'),
            'disp_tribunal_turnos': pd.read_excel(excel_path, sheet_name='Disponibilidad-tribunal-turnos'),
            'disp_alumnos_tribunal': pd.read_excel(excel_path, sheet_name='Disponibilidad-alumnos-tribunal')
        }
        
        # Obtener dimensiones del problema
        self.num_students = len(self.excel_data['disp_alumnos_turnos'])
        self.num_timeslots = len(self.excel_data['turnos'])
        self.num_professors = len(self.excel_data['disp_tribunal_turnos'])
        
        # Mapeos de identificadores
        self.tribunal_ids = self.excel_data['disp_alumnos_tribunal'].columns[1:]
        self.tribunal_index_to_id = dict(enumerate(self.tribunal_ids))
        self.tribunal_id_to_index = {v: k for k, v in self.tribunal_index_to_id.items()}
        
        self.timeslot_ids = self.excel_data['disp_tribunal_turnos'].columns[1:]
        self.timeslot_index_to_id = dict(enumerate(self.timeslot_ids))
        self.timeslot_id_to_index = {v: k for k, v in self.timeslot_index_to_id.items()}

    def calculate_fitness(self, solution: TimeTableSolution) -> float:
        """
        Calcula el valor de aptitud de una solución dada.
        En principio se dan puntauciones muy altas cuando se cumplen las restricciones, y
        penalizaciones bastante severas cuando no se cumplen.
        El valor de normalización hace que cualquier solución que cumpla las restricciones
        tenga una buena puntuación, realmente la solución si cumple las restricciones sería buena.
        En este caso se hace dificil diferenciar soluciones buenas de soluciones mejores, debido
        a que las restricciones no son muchas
        
        Args:
            solution (TimeTableSolution): Solución a evaluar
        
        Returns:
            float: Valor de aptitud normalizado
        """
        total_score = 0
        penalties = 0
        used_timeslots = {}
        
        for student in range(self.num_students):
            timeslot = int(solution.chromosome[student, 0])
            tribunal = [int(x) for x in solution.chromosome[student, 1:]]
            # --- ESTRUCTURA DE PUNTACION ---
            # Verificar disponibilidad de estudiante
            # Por cada estudiante se suma 1 punto en caso de estar disponible
            if self.excel_data['disp_alumnos_turnos'].iloc[student, timeslot + 1] == 1:
                total_score += 1
            else:
                penalties += 10
            
            # Verificar disponibilidad de profesores
            # Por cada profesor se suma 1 punto en caso de estar disponible
            for prof in tribunal:
                if self.excel_data['disp_tribunal_turnos'].iloc[prof, timeslot + 1] == 1:
                    total_score += 1
                else:
                    penalties += 5
            
            # Verificar conflictos de horarios
            # Penalización por conflictos de horario
            if timeslot in used_timeslots:
                penalties += 50
            else:
                used_timeslots[timeslot] = student
            
            # Verificar conflictos de tribunal
            # Penalización por conflictos de tribunal (profesores)
            for other_student in range(self.num_students):
                if other_student != student and int(solution.chromosome[other_student, 0]) == timeslot:
                    common_profs = set(tribunal) & set(int(x) for x in solution.chromosome[other_student, 1:])
                    penalties += len(common_profs) * 10
        
        # Sistema de NORMALIZACIÓN: normalizar puntuación, de cara a que cuando se encuentre con problemas
        # más generales se proporcione el gradiente necesario para una evolución más prolongada hacia 
        # soluciones óptimas.
        max_score = self.num_students * 4
        '''
        # Esta normalización hace que se encuentre muy rápido una solución óptima
        normalized_score = total_score / max_score
        normalized_penalties = min(penalties / max_score, 1)
        '''
        # Con esta normalización se obtiene más diversidad en la población
        fitness_range = max_score * 0.5  # Crear más espacio entre soluciones
        normalized_score = (total_score / max_score) * fitness_range
        normalized_penalties = (penalties / max_score) * fitness_range
                
        return normalized_score - normalized_penalties
    
    def check_feasibility(self, solution: TimeTableSolution) -> Tuple[bool, str]:
        """
        Verifica si una solución es factible según las restricciones del problema.
        
        Args:
            solution (TimeTableSolution): Solución a verificar
        
        Returns:
            Tuple[bool, str]: (es_factible, mensaje_error)
        """
        # Verificar estudiantes y turnos
        for estudiante in range(self.num_students):
            nombre_estudiante = self.excel_data['disp_alumnos_turnos'].iloc[estudiante, 0]
            turno = int(solution.chromosome[estudiante, 0])
            tribunal = [int(x) for x in solution.chromosome[estudiante, 1:]]
            
            if self.excel_data['disp_alumnos_turnos'].iloc[estudiante, turno + 1] != 1:
                return False, f"Estudiante {nombre_estudiante} asignado a turno no disponible"
            
            for profesor in tribunal:
                if self.excel_data['disp_alumnos_tribunal'].iloc[estudiante, profesor + 1] != 1:
                    return False, f"Profesor {self.tribunal_index_to_id[profesor]} no disponible para estudiante {nombre_estudiante}"
                
                if self.excel_data['disp_tribunal_turnos'].iloc[profesor, turno + 1] != 1:
                    return False, f"Profesor {self.tribunal_index_to_id[profesor]} no disponible en turno {self.timeslot_index_to_id[turno]}"
            
            if len(set(tribunal)) != 3:
                return False, f"Tribunal del estudiante {nombre_estudiante} tiene profesores duplicados"
        
        return True, "Solución factible"

    def export_solution(self, solution: TimeTableSolution, excel_path: str):
        """
        Exporta una solución a un archivo Excel.
        
        Args:
            solution (TimeTableSolution): Solución a exportar
            excel_path (str): Ruta donde guardar el archivo Excel
        """
        try:
            # Crear DataFrame para horarios
            best_horario = pd.DataFrame(columns=['Alumno', 'Horario', 'Tribunal1', 'Tribunal2', 'Tribunal3'])
            best_horario['Alumno'] = self.excel_data['disp_alumnos_turnos'].iloc[:, 0]
            best_horario['Horario'] = [self.timeslot_index_to_id[idx] for idx in solution.chromosome[:, 0]]
            
            for i in range(3):
                best_horario[f'Tribunal{i+1}'] = [self.tribunal_index_to_id[idx] for idx in solution.chromosome[:, i+1]]
            
            # Crear DataFrame para asignación de tribunales por turno
            best_tribunal_turnos = pd.DataFrame(
                index=self.excel_data['disp_alumnos_turnos'].iloc[:, 0],
                columns=self.timeslot_ids
            )
            
            for student in range(self.num_students):
                timeslot_id = self.timeslot_index_to_id[solution.chromosome[student, 0]]
                tribunal = [self.tribunal_index_to_id[idx] for idx in solution.chromosome[student, 1:]]
                tribunal_str = f"{tribunal[0]},{tribunal[1]},{tribunal[2]}"
                best_tribunal_turnos.loc[best_tribunal_turnos.index[student], timeslot_id] = tribunal_str
            
            # Guardar en Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                best_horario.to_excel(writer, sheet_name='best_horario', index=False)
                best_tribunal_turnos.to_excel(writer, sheet_name='best_tribunal_turnos')
            print(f"Solución exportada exitosamente a: {excel_path}")
            
        except Exception as e:
            print(f"Error al exportar la solución: {str(e)}")

    def analyze_problem_constraints(self):
        """
        Analiza las restricciones del problema para verificar su factibilidad.
        
        Raises:
            ValueError: Si se detectan problemas de factibilidad
        """
        problemas = []
        for estudiante in range(self.num_students):
            nombre = self.excel_data['disp_alumnos_turnos'].iloc[estudiante, 0]
            slots_disponibles = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[estudiante, 1:] == 1)
            profs_disponibles = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[estudiante, 1:] == 1)
            
            if slots_disponibles == 0:
                problemas.append(f"Estudiante {nombre} no tiene turnos disponibles")
            if profs_disponibles < 3:
                problemas.append(f"Estudiante {nombre} solo tiene {profs_disponibles} tribunales disponibles")
            elif profs_disponibles < 4:
                print(f"Advertencia: Estudiante {nombre} tiene opciones muy limitadas ({profs_disponibles} tribunales)")
        
        if problemas:
            raise ValueError("Problemas de factibilidad detectados:\n" + "\n".join(problemas))