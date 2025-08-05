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
from datetime import datetime, date
from typing import Dict, List, Tuple
from data_structures import TimeTableSolution

class TimetablingProblem:
    """
    Clase base que define la estructura y funcionalidad común para resolver
    el problema de asignación de tribunales y horarios.
    """
    def __init__(self, excel_path: str):
        """
        Inicializa el problema de timetabling cargando los datos necesarios.
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
        
        # Crear diccionarios de mapeo usando índices numéricos
        self.timeslot_dates = {}  # índice -> fecha
        self.timeslot_to_id = {}  # índice -> ID
        self.id_to_timeslot = {}  # ID -> índice
        
        # Obtener los IDs de profesores y turnos
        self.tribunal_ids = self.excel_data['disp_alumnos_tribunal'].columns[1:]
        self.timeslot_ids = self.excel_data['disp_tribunal_turnos'].columns[1:]
        
        # Crear mapeos de tribunales
        self.tribunal_index_to_id = dict(enumerate(self.tribunal_ids))
        self.tribunal_id_to_index = {v: k for k, v in self.tribunal_index_to_id.items()}
        
        # Crear mapeos usando índices numéricos simples
        turnos_df = self.excel_data['turnos']
        for idx in range(self.num_timeslots):
            fecha = pd.to_datetime(turnos_df.iloc[idx]['Fecha']).date()
            turno_id = turnos_df.iloc[idx]['Turno']
            
            self.timeslot_dates[idx] = fecha
            self.timeslot_to_id[idx] = turno_id
            self.id_to_timeslot[turno_id] = idx
        
        print("\nMapeos creados:")
        print(f"Total timeslots: {self.num_timeslots}")
        print(f"Índices válidos: {sorted(self.timeslot_dates.keys())}")
    
    def calculate_fitness(self, solution: TimeTableSolution) -> float:
        """
        Calcula el valor de aptitud de una solución incluyendo restricciones duras y blandas (suaves)
        
        """
        #print("\nIniciando cálculo de fitness")
        #print(f"Dimensiones de la solución: {solution.chromosome.shape}")
        #print(f"Contenido del cromosoma:\n{solution.chromosome}")
        #print(f"Índices de timeslot válidos: {list(range(self.num_timeslots))}")
        
        # Validación preliminar: verifica que la asignación de horarios estén dentro de los límites permitidos
        # En caso de encontrar un horario fuera de rango se devolvería un valor de fitness=0.0
        max_timeslot = np.max(solution.chromosome[:, 0])
        if max_timeslot >= self.num_timeslots:
            print(f"Error: solución contiene timeslot {max_timeslot} fuera de rango (max válido: {self.num_timeslots-1})")
            return 0.0
        
        # Inicializaciones
        total_score = 0 # Almacena la puntuación positiva, se obtiene cumpliendo restricciones
        penalties = 0   # Almacena la penalizaciones cuando se incumplen restricciones
        used_timeslots = {} # Almacena los horarios ya asignados, así se evitan conflictos
        prof_assignments = {} # Almacena las asignaciones de profesores por fechas
        
        # Evaluamos las restricciones duras
        # - Disponibilidad de horarios de estudiantes
        # - Disponibilidad de miembros del tribunal
        # - Conflictos de horarios
        for student in range(self.num_students):
            timeslot = int(solution.chromosome[student, 0])
            #print(f"\nProcesando estudiante {student}, timeslot {timeslot}")
            
            tribunal = [int(x) for x in solution.chromosome[student, 1:]]
            
            # --- RESTRICCIONES DURAS ---
            # Si un estudiante está disponible en ese horario se suma al total_score
            if self.excel_data['disp_alumnos_turnos'].iloc[student, timeslot + 1] == 1:
                total_score += 1
            else:
                # Si no está disponible de añade una penalización
                penalties += 10
            
            # Lo mismo con un profesor, pero con menor penalización, por ser menos crítica
            for prof in tribunal:
                if self.excel_data['disp_tribunal_turnos'].iloc[prof, timeslot + 1] == 1:
                    total_score += 1
                else:
                    penalties += 5
            
            # Penalizaciones para conflictos de horarios:
            # - Un mismo horario no puede asignarse a más de un estudiante
            # - No puede haber profesores en tribunales diferentes en un mismo horario
            if timeslot in used_timeslots:
                penalties += 50
            else:
                used_timeslots[timeslot] = student
            
            for other_student in range(self.num_students):
                if other_student != student and int(solution.chromosome[other_student, 0]) == timeslot:
                    common_profs = set(tribunal) & set(int(x) for x in solution.chromosome[other_student, 1:])
                    penalties += len(common_profs) * 10
            
            # --- RESTRICCIONES SUAVES ---
            fecha = self.timeslot_dates[timeslot] # Obtener la fecha del Timeslot asignado
            for prof in tribunal: # Recorremos los profesores asignados al tribunal actual
                # Si el profesor no esta registrado en el diccionario se crea un registro para el
                # prof_assignments[prof]: Es un subdiccionario donde las claves son fechas (fecha) y 
                # los valores son listas de timeslots asignados a ese profesor en ese día
                if prof not in prof_assignments: 
                    prof_assignments[prof] = {}
                # Inicializmos el registro del día
                if fecha not in prof_assignments[prof]:
                    # Si la fecha no está registrada para ese profesor, 
                    # se crea una una entrada vacia para esa fecha
                    prof_assignments[prof][fecha] = [] 
                # Registramos el Timeslot asignado
                prof_assignments[prof][fecha].append(timeslot)
        
        # Evaluar restricciones suaves por profesor
        for prof, date_assignments in prof_assignments.items():
            total_days = len(date_assignments)
            total_tribunals = sum(len(slots) for slots in date_assignments.values())
            
            # Bonificación por eficiencia en días
            efficiency_bonus = (total_tribunals / total_days) if total_days > 0 else 0
            total_score += efficiency_bonus * 0.5
            
            # Evaluar cada día del profesor
            for fecha, turnos in date_assignments.items():
                num_turnos = len(turnos)
                
                # Bonificación por múltiples tribunales en un día
                if num_turnos > 1:
                    total_score += num_turnos * 0.5
                    
                    # Penalización por huecos entre tribunales
                    turnos_ordenados = sorted(turnos)
                    for i in range(1, len(turnos_ordenados)):
                        gap = turnos_ordenados[i] - turnos_ordenados[i-1]
                        if gap > 1:  # Si hay hueco entre tribunales
                            penalties += (gap - 1) * 0.3
                            
                # Bonificación por día completo
                if num_turnos >= 3:
                    total_score += 1.0
        
        # Normalización
        # Máximo teórico de puntuación: 
        # - 1 punto por disponibilidad en ese horario
        # - 3 puntos (1 por cada miembro del tribunal)
        # Lo que hace un total de 4
        max_score = self.num_students * 4 
        # hacemos una escala del rango de fitness, de manera que permanezca contenido
        fitness_range = max_score * 0.5
        # Normalizamos la puntuación total: escalada al rango definido por fitness_range
        normalized_score = (total_score / max_score) * fitness_range
        # Normalizamos las penalizaciones, en la misma escala que la puntuacion total
        normalized_penalties = (penalties / max_score) * fitness_range
        # Cálculo del fitness total
        final_fitness = normalized_score - normalized_penalties
        #print(f"Fitness calculado: {final_fitness}")
        return final_fitness
    
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
            # Obtener identificadores de turnos directamente de los datos
            timeslot_ids = self.excel_data['turnos']['Turno'].tolist()
            
            # Crear DataFrame para horarios
            best_horario = pd.DataFrame(columns=['Alumno', 'Horario', 'Tribunal1', 'Tribunal2', 'Tribunal3'])
            
            # Obtener nombres de alumnos
            best_horario['Alumno'] = self.excel_data['disp_alumnos_turnos'].iloc[:, 0]
            
            # Asignar horarios usando los IDs de turno
            best_horario['Horario'] = [timeslot_ids[int(idx)] for idx in solution.chromosome[:, 0]]
            
            # Obtener nombres de profesores
            profesor_ids = self.excel_data['disp_tribunal_turnos'].iloc[:, 0].tolist()
            
            # Asignar tribunales usando los IDs de profesor
            for i in range(3):
                best_horario[f'Tribunal{i+1}'] = [profesor_ids[int(idx)] for idx in solution.chromosome[:, i+1]]
            
            # Crear DataFrame para asignación de tribunales por turno
            index = self.excel_data['disp_alumnos_turnos'].iloc[:, 0]
            columns = timeslot_ids
            best_tribunal_turnos = pd.DataFrame(index=index, columns=columns)
            
            # Llenar la matriz de asignaciones
            for student in range(self.num_students):
                slot_idx = int(solution.chromosome[student, 0])
                slot_id = timeslot_ids[slot_idx]
                
                tribunal = [profesor_ids[int(idx)] for idx in solution.chromosome[student, 1:]]
                tribunal_str = ",".join(tribunal)
                
                best_tribunal_turnos.loc[index[student], slot_id] = tribunal_str
            
            # Guardar en Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                best_horario.to_excel(writer, sheet_name='best_horario', index=False)
                best_tribunal_turnos.to_excel(writer, sheet_name='best_tribunal_turnos')
            
            print(f"Solución exportada exitosamente a: {excel_path}")
            
        except Exception as e:
            print(f"Error al exportar la solución: {str(e)}")
            raise

    def analyze_schedule_metrics(self, solution: TimeTableSolution) -> Dict[str, float]:
        """
        Analiza métricas específicas de la distribución de horarios.
        
        Args:
            solution (TimeTableSolution): Solución a analizar
        
        Returns:
            Dict[str, float]: Diccionario con métricas detalladas de la solución
        """
        metrics = {
            'avg_tribunals_per_day': 0.0,
            'total_gaps': 0,
            'days_with_single_tribunal': 0,
            'days_fully_utilized': 0,
            'prof_day_efficiency': 0.0,
            'max_tribunals_in_day': 0,
            'total_different_days': 0
        }
        
        prof_assignments: Dict[int, Dict[datetime.date, List[int]]] = {}
        
        # Recopilar datos
        for student in range(self.num_students):
            timeslot = int(solution.chromosome[student, 0])
            tribunal = [int(x) for x in solution.chromosome[student, 1:]]
            fecha = self.timeslot_dates[timeslot]
            
            for prof in tribunal:
                if prof not in prof_assignments:
                    prof_assignments[prof] = {}
                if fecha not in prof_assignments[prof]:
                    prof_assignments[prof][fecha] = []
                prof_assignments[prof][fecha].append(timeslot)
        
        # Calcular métricas
        total_days = 0
        total_tribunals = 0
        total_gaps = 0
        days_single = 0
        days_full = 0
        max_tribunals = 0
        
        for prof, date_assignments in prof_assignments.items():
            prof_days = len(date_assignments)
            total_days += prof_days
            
            for fecha, turnos in date_assignments.items():
                num_turnos = len(turnos)
                total_tribunals += num_turnos
                max_tribunals = max(max_tribunals, num_turnos)
                
                if num_turnos == 1:
                    days_single += 1
                elif num_turnos >= 3:
                    days_full += 1
                
                # Contar huecos
                turnos_ordenados = sorted(turnos)
                for i in range(1, len(turnos_ordenados)):
                    gap = turnos_ordenados[i] - turnos_ordenados[i-1]
                    if gap > 1:
                        total_gaps += gap - 1
        
        num_profs = len(prof_assignments)
        if num_profs > 0 and total_days > 0:
            metrics['avg_tribunals_per_day'] = total_tribunals / total_days
            metrics['total_gaps'] = total_gaps
            metrics['days_with_single_tribunal'] = days_single
            metrics['days_fully_utilized'] = days_full
            metrics['prof_day_efficiency'] = total_tribunals / (num_profs * len(set(self.timeslot_dates.values())))
            metrics['max_tribunals_in_day'] = max_tribunals
            metrics['total_different_days'] = total_days
        
        return metrics

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