"""
Módulo que implementa una versión mejorada del algoritmo Harmony Search para la 
asignación de tribunales.

Este módulo extiende la clase base TimetablingProblem para implementar una
solución basada en el algoritmo Harmony Search al problema de asignación de tribunales.

Autor: Juan José Jiménez González
Institución: Universidad Isabel I
Fecha: 2024
"""
import random
import numpy as np
import logging
import sys
from typing import List, Tuple, Optional, Dict
from timetabling_problem import TimetablingProblem
from data_structures import TimeTableSolution

def setup_logging(name: str) -> logging.Logger:
    """
    Configura el sistema de logging para el algoritmo.
    
    Args:
        name: Nombre del logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Evitar duplicación de handlers
    if not logger.handlers:
        # Crear formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

class TimetablingHS(TimetablingProblem):
    """
    Implementación mejorada del algoritmo Harmony Search para resolver el problema de timetabling.
    """

    def __init__(self, excel_path: str):
        super().__init__(excel_path)
        
        # Parámetros del algoritmo ajustados para problemas pequeños
        self.hms = max(5, min(20, self.num_students))  # HMS adaptativo
        self.hmcr_init = 0.85
        self.par_init = 0.3
        self.max_iterations = 200
        self.max_iterations_without_improvement = 20
        self.min_par = 0.2
        self.max_par = 0.4
        self.min_hmcr = 0.75
        self.max_hmcr = 0.95
        self.min_generations = 15
        self.local_search_probability = 0.2
        self.min_fitness_threshold = 0.2  # Umbral reducido
        self.similarity_threshold = 0.6
        
        # Configuración de logging
        self.logger = setup_logging('TimetablingHS')

    def __init__old(self, excel_path: str):
        """
        Inicializa el algoritmo HS con los parámetros específicos mejorados.
        
        Args:
            excel_path (str): Ruta al archivo Excel con los datos de entrada
        """
        super().__init__(excel_path)  # Llamada al constructor de TimetablingProblem
      
        # Primero inicializar los parámetros propios
        self.hms = 30  
        self.hmcr_init = 0.7  
        self.par_init = 0.3  
        self.max_iterations = 2000  
        self.max_iterations_without_improvement = 100  
        self.min_par = 0.1  
        self.max_par = 0.4  
        self.min_hmcr = 0.7  
        self.max_hmcr = 0.95  
        self.min_generations = 50  
        self.local_search_probability = 0.3  
        self.diversification_frequency = 100  
        self.min_fitness_threshold = 0.7  
    
        # Después llamar al constructor de la clase base
        super().__init__(excel_path)

    def _get_ordered_students(self) -> List[int]:
        """
        Ordena los estudiantes según sus restricciones.
        
        Returns:
            List[int]: Lista de índices de estudiantes ordenados por número de restricciones
                    (de más restrictivo a menos restrictivo)
        """
        student_restrictions = []
        for student in range(self.num_students):
            # Contar slots disponibles
            slots = np.sum(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)
            # Contar profesores disponibles
            profs = np.sum(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)
            # El producto de ambos nos da una medida de la flexibilidad
            # Cuanto menor sea, más restrictivo es el estudiante
            student_restrictions.append((student, slots * profs))
        
        # Ordenar por restricciones (menor a mayor flexibilidad)
        return [s[0] for s in sorted(student_restrictions, key=lambda x: x[1])]

    def _validate_student_assignment(self, student: int, slot: int, tribunal: List[int]) -> bool:
        """
        Valida la asignación de un estudiante a un slot y tribunal específicos.
        """
        # Verificar disponibilidad del estudiante en el slot
        if self.excel_data['disp_alumnos_turnos'].iloc[student, slot + 1] != 1:
            self.logger.debug(f"Estudiante {student} no disponible en slot {slot}")
            return False
            
        # Verificar disponibilidad de profesores
        for prof in tribunal:
            if self.excel_data['disp_tribunal_turnos'].iloc[prof, slot + 1] != 1:
                self.logger.debug(f"Profesor {prof} no disponible en slot {slot}")
                return False
            if self.excel_data['disp_alumnos_tribunal'].iloc[student, prof + 1] != 1:
                self.logger.debug(f"Profesor {prof} no compatible con estudiante {student}")
                return False
                
        return True

    def generate_random_harmony(self) -> Optional[TimeTableSolution]:
        """
        Genera una nueva armonía aleatoria con mejor manejo de restricciones y diagnóstico.
        """
        max_attempts_per_student = 30
        chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        # Diagnóstico inicial
        self.logger.debug(f"Generando solución para {self.num_students} estudiantes")
        self.logger.debug(f"Slots disponibles: {self.num_timeslots}")
        
        for student in range(self.num_students):
            attempts = 0
            assigned = False
            
            while attempts < max_attempts_per_student and not assigned:
                # Obtener slots disponibles
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                if not available_slots:
                    self.logger.debug(f"No hay slots disponibles para estudiante {student}")
                    attempts += 1
                    continue
                
                # Seleccionar slot aleatorio
                slot = random.choice(available_slots)
                
                # Obtener profesores disponibles
                available_profs = []
                for prof in range(self.num_professors):
                    if (self.excel_data['disp_alumnos_tribunal'].iloc[student, prof + 1] == 1 and
                        self.excel_data['disp_tribunal_turnos'].iloc[prof, slot + 1] == 1):
                        available_profs.append(prof)
                
                if len(available_profs) >= 3:
                    tribunal = random.sample(available_profs, 3)
                    if self._validate_student_assignment(student, slot, tribunal):
                        chromosome[student, 0] = slot
                        chromosome[student, 1:] = tribunal
                        used_timeslots.add(slot)
                        assigned = True
                        self.logger.debug(f"Estudiante {student} asignado exitosamente")
                else:
                    self.logger.debug(f"Insuficientes profesores disponibles para estudiante {student}")
                
                attempts += 1
            
            if not assigned:
                self.logger.debug(f"No se pudo asignar estudiante {student} después de {attempts} intentos")
                return None
        
        # Calcular fitness y validar solución
        solution = TimeTableSolution(chromosome=chromosome)
        solution.fitness = self.calculate_fitness(solution)
        
        if solution.fitness > self.min_fitness_threshold:
            self.logger.debug(f"Solución generada con fitness {solution.fitness}")
            return solution
        else:
            self.logger.debug(f"Solución descartada por bajo fitness {solution.fitness}")
            return None

    def fast_local_search(self, solution: TimeTableSolution) -> TimeTableSolution:
        """
        Realiza una búsqueda local rápida para mejorar una solución dada.
        
        Args:
            solution: Solución a mejorar
            
        Returns:
            TimeTableSolution: Solución mejorada
        """
        improved = True
        best_fitness = solution.fitness
        best_chromosome = solution.chromosome.copy()
        max_attempts = 5
        
        while improved and max_attempts > 0:
            improved = False
            max_attempts -= 1
            
            # Seleccionar estudiantes aleatoriamente para optimización local
            num_students_to_modify = min(3, self.num_students)
            students = random.sample(range(self.num_students), num_students_to_modify)
            
            for student in students:
                original_assignment = best_chromosome[student].copy()
                used_timeslots = {best_chromosome[i, 0] for i in range(self.num_students) if i != student}
                
                # Encontrar slots disponibles
                available_slots = []
                for slot in range(self.num_timeslots):
                    if (slot not in used_timeslots and 
                        self.excel_data['disp_alumnos_turnos'].iloc[student, slot + 1] == 1):
                        available_slots.append(slot)
                
                if available_slots:
                    # Probar diferentes slots
                    for slot in random.sample(available_slots, min(3, len(available_slots))):
                        best_chromosome[student, 0] = slot
                        
                        # Encontrar profesores disponibles
                        available_profs = []
                        for prof in range(self.num_professors):
                            if (self.excel_data['disp_alumnos_tribunal'].iloc[student, prof + 1] == 1 and
                                self.excel_data['disp_tribunal_turnos'].iloc[prof, slot + 1] == 1):
                                available_profs.append(prof)
                        
                        if len(available_profs) >= 3:
                            # Probar diferentes combinaciones de tribunal
                            max_combinations = min(5, len(available_profs) * (len(available_profs) - 1) * (len(available_profs) - 2) // 6)
                            for _ in range(max_combinations):
                                tribunal = random.sample(available_profs, 3)
                                best_chromosome[student, 1:] = tribunal
                                
                                temp_solution = TimeTableSolution(chromosome=best_chromosome.copy())
                                temp_solution.fitness = self.calculate_fitness(temp_solution)
                                
                                if temp_solution.fitness > best_fitness:
                                    best_fitness = temp_solution.fitness
                                    improved = True
                                    self.logger.debug(f"Mejora encontrada: {best_fitness:.4f}")
                                    break
                        
                        if improved:
                            break
                        
                    if not improved:
                        best_chromosome[student] = original_assignment
        
        return TimeTableSolution(chromosome=best_chromosome, fitness=best_fitness)

    def generate_initial_harmony_memory(self) -> List[TimeTableSolution]:
        """
        Genera la memoria armónica inicial con mejor manejo de errores y diagnóstico.
        """
        harmony_memory = []
        max_total_attempts = self.hms * 30
        current_attempts = 0
        
        self.logger.info(f"Iniciando generación de memoria armónica (HMS={self.hms})")
        self.logger.info(f"Umbral de fitness mínimo: {self.min_fitness_threshold}")
        
        while len(harmony_memory) < self.hms and current_attempts < max_total_attempts:
            try:
                harmony = self.generate_random_harmony()
                if harmony is not None:
                    harmony_memory.append(harmony)
                    self.logger.info(f"Generada solución {len(harmony_memory)}/{self.hms} "
                                   f"(fitness={harmony.fitness:.4f})")
            except Exception as e:
                self.logger.error(f"Error generando solución: {str(e)}")
            
            current_attempts += 1
            
            # Reducir umbral si no se encuentran soluciones
            if current_attempts % 10 == 0 and not harmony_memory:
                self.min_fitness_threshold *= 0.9
                self.logger.info(f"Reduciendo umbral de fitness a {self.min_fitness_threshold}")
        
        if not harmony_memory:
            self.logger.error("No se pudo generar ninguna solución válida")
            raise ValueError("No se pudo generar una memoria inicial válida")
        
        self.logger.info(f"Memoria inicial generada con {len(harmony_memory)} soluciones")
        return sorted(harmony_memory, key=lambda x: x.fitness, reverse=True)

    def maintain_diversity(self, harmony_memory: List[TimeTableSolution]) -> List[TimeTableSolution]:
        """
        Mantiene la diversidad en la memoria armónica.
        
        Args:
            harmony_memory: Lista actual de soluciones
            
        Returns:
            Lista de soluciones con diversidad mejorada
        """
        unique_solutions = {}
        for sol in harmony_memory:
            hash_key = hash(tuple(sol.chromosome.flatten()))
            if hash_key not in unique_solutions or sol.fitness > unique_solutions[hash_key].fitness:
                unique_solutions[hash_key] = sol
        
        diverse_memory = list(unique_solutions.values())
        
        while len(diverse_memory) < self.hms:
            new_solution = self._generate_single_harmony()
            if new_solution is not None:
                new_solution.fitness = self.calculate_fitness(new_solution)
                if new_solution.fitness > self.min_fitness_threshold:
                    hash_key = hash(tuple(new_solution.chromosome.flatten()))
                    if hash_key not in unique_solutions:
                        diverse_memory.append(new_solution)
                        unique_solutions[hash_key] = new_solution
        
        return sorted(diverse_memory, key=lambda x: x.fitness, reverse=True)

    def _calculate_similarity(self, sol1: TimeTableSolution, sol2: TimeTableSolution) -> float:
        """
        Calcula la similitud entre dos soluciones.
        
        Args:
            sol1, sol2: Soluciones a comparar
            
        Returns:
            Medida de similitud entre 0 y 1
        """
        same_timeslots = np.sum(sol1.chromosome[:, 0] == sol2.chromosome[:, 0])
        same_tribunal_members = 0
        
        for i in range(self.num_students):
            tribunal1 = set(sol1.chromosome[i, 1:])
            tribunal2 = set(sol2.chromosome[i, 1:])
            same_tribunal_members += len(tribunal1.intersection(tribunal2))
        
        return (same_timeslots + same_tribunal_members / 3) / (self.num_students * 2)

    def adjust_parameters(self, iteration: int, best_fitness: float) -> Tuple[float, float]:
        """
        Ajusta dinámicamente los parámetros HMCR y PAR.
        
        Args:
            iteration: Iteración actual
            best_fitness: Mejor fitness actual
            
        Returns:
            Tupla con nuevos valores de HMCR y PAR
        """
        # Factor de progreso
        progress = iteration / self.max_iterations
        
        # Factor de calidad
        quality_factor = 1 - best_fitness
        
        # Ajustar HMCR
        hmcr = self.min_hmcr + (self.max_hmcr - self.min_hmcr) * (progress + quality_factor) / 2
        
        # Ajustar PAR
        par = self.max_par - (self.max_par - self.min_par) * (progress - quality_factor) / 2
        
        # Asegurar límites
        hmcr = max(self.min_hmcr, min(self.max_hmcr, hmcr))
        par = max(self.min_par, min(self.max_par, par))
        
        return hmcr, par

    def extended_local_search(self, solution: TimeTableSolution) -> TimeTableSolution:
        """
        Realiza una búsqueda local más exhaustiva.
        """
        improved = True
        best_fitness = solution.fitness
        best_chromosome = solution.chromosome.copy()
        search_iterations = 0
        max_search_iterations = 20
        
        while improved and search_iterations < max_search_iterations:
            improved = False
            search_iterations += 1
            
            # Intentar mejoras por estudiante
            for student in range(self.num_students):
                original_assignment = best_chromosome[student].copy()
                used_timeslots = {best_chromosome[i, 0] for i in range(self.num_students) if i != student}
                
                # Probar cambios de horario
                available_slots = np.where(self.excel_data['disp_alumnos_turnos'].iloc[student, 1:] == 1)[0]
                available_slots = [slot for slot in available_slots if slot not in used_timeslots]
                
                for slot in available_slots:
                    best_chromosome[student, 0] = slot
                    
                    # Probar diferentes combinaciones de tribunal
                    available_profs = np.where(self.excel_data['disp_alumnos_tribunal'].iloc[student, 1:] == 1)[0]
                    if len(available_profs) >= 3:
                        for _ in range(5):  # Intentar varias combinaciones
                            best_chromosome[student, 1:] = np.random.choice(available_profs, 3, replace=False)
                            temp_solution = TimeTableSolution(chromosome=best_chromosome.copy())
                            temp_solution.fitness = self.calculate_fitness(temp_solution)
                            
                            if temp_solution.fitness > best_fitness:
                                best_fitness = temp_solution.fitness
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if not improved:
                    best_chromosome[student] = original_assignment
                else:
                    break  # Continuar con el siguiente ciclo de mejora
        
        return TimeTableSolution(chromosome=best_chromosome, fitness=best_fitness)

    def solve(self) -> Tuple[TimeTableSolution, List[float]]:
        """
        Ejecuta el algoritmo Harmony Search con mejor diagnóstico y manejo de errores.
        """
        try:
            self.logger.info("Iniciando algoritmo Harmony Search")
            self.logger.info(f"Dimensiones del problema:")
            self.logger.info(f"Estudiantes: {self.num_students}")
            self.logger.info(f"Profesores: {self.num_professors}")
            self.logger.info(f"Turnos: {self.num_timeslots}")
            
            harmony_memory = self.generate_initial_harmony_memory()
            best_solution = harmony_memory[0]
            best_fitness_history = [best_solution.fitness]
            iterations_without_improvement = 0
            
            for iteration in range(self.max_iterations):
                try:
                    hmcr, par = self.adjust_parameters(iteration, best_solution.fitness)
                    new_harmony = self.improvise_new_harmony(harmony_memory, hmcr, par)
                    
                    if random.random() < self.local_search_probability:
                        new_harmony = self.fast_local_search(new_harmony)
                    
                    worst_harmony = min(harmony_memory, key=lambda x: x.fitness)
                    if new_harmony.fitness > worst_harmony.fitness:
                        harmony_memory.remove(worst_harmony)
                        harmony_memory.append(new_harmony)
                        
                        if new_harmony.fitness > best_solution.fitness:
                            best_solution = new_harmony
                            iterations_without_improvement = 0
                            self.logger.info(f"Nueva mejor solución en iteración {iteration}: "
                                           f"fitness = {best_solution.fitness:.4f}")
                        else:
                            iterations_without_improvement += 1
                    else:
                        iterations_without_improvement += 1
                    
                    best_fitness_history.append(best_solution.fitness)
                    
                    # Criterios de parada
                    if (iteration >= self.min_generations and 
                        iterations_without_improvement >= self.max_iterations_without_improvement):
                        self.logger.info(f"Convergencia detectada en iteración {iteration}")
                        break
                    
                    if iteration % 50 == 0:
                        self.logger.info(f"Iteración {iteration}: "
                                       f"Mejor fitness = {best_solution.fitness:.4f}, "
                                       f"HMCR = {hmcr:.3f}, PAR = {par:.3f}")
                        
                except Exception as e:
                    self.logger.error(f"Error en iteración {iteration}: {str(e)}")
                    continue
            
            return best_solution, best_fitness_history
            
        except Exception as e:
            self.logger.error(f"Error en solve(): {str(e)}")
            raise
    
    def improvise_new_harmony(self, harmony_memory: List[TimeTableSolution], 
                         hmcr: float, par: float) -> TimeTableSolution:
        """
        Improvisa una nueva armonía basada en la memoria existente.
        
        Args:
            harmony_memory: Lista de armonías actuales
            hmcr: Harmony Memory Considering Rate
            par: Pitch Adjustment Rate
            
        Returns:
            TimeTableSolution: Nueva armonía
        """
        new_chromosome = np.zeros((self.num_students, 4), dtype=int)
        used_timeslots = set()
        
        for student in range(self.num_students):
            if random.random() < hmcr:
                # Usar memoria
                weights = [h.fitness for h in harmony_memory]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    selected_harmony = random.choices(harmony_memory, weights=weights, k=1)[0]
                else:
                    selected_harmony = random.choice(harmony_memory)
                
                new_chromosome[student] = selected_harmony.chromosome[student].copy()
                
                if random.random() < par:
                    # Ajuste de tono
                    available_slots = []
                    for slot in range(self.num_timeslots):
                        if (slot not in used_timeslots and 
                            self.excel_data['disp_alumnos_turnos'].iloc[student, slot + 1] == 1):
                            available_slots.append(slot)
                    
                    if available_slots:
                        new_chromosome[student, 0] = random.choice(available_slots)
                        
                        # Obtener profesores disponibles
                        available_profs = []
                        for prof in range(self.num_professors):
                            if (self.excel_data['disp_alumnos_tribunal'].iloc[student, prof + 1] == 1 and
                                self.excel_data['disp_tribunal_turnos'].iloc[prof, new_chromosome[student, 0] + 1] == 1):
                                available_profs.append(prof)
                        
                        if len(available_profs) >= 3:
                            new_chromosome[student, 1:] = random.sample(available_profs, 3)
            else:
                # Generación aleatoria
                available_slots = []
                for slot in range(self.num_timeslots):
                    if (slot not in used_timeslots and 
                        self.excel_data['disp_alumnos_turnos'].iloc[student, slot + 1] == 1):
                        available_slots.append(slot)
                
                if available_slots:
                    new_chromosome[student, 0] = random.choice(available_slots)
                    
                    # Obtener profesores disponibles
                    available_profs = []
                    for prof in range(self.num_professors):
                        if (self.excel_data['disp_alumnos_tribunal'].iloc[student, prof + 1] == 1 and
                            self.excel_data['disp_tribunal_turnos'].iloc[prof, new_chromosome[student, 0] + 1] == 1):
                            available_profs.append(prof)
                    
                    if len(available_profs) >= 3:
                        new_chromosome[student, 1:] = random.sample(available_profs, 3)
            
            used_timeslots.add(new_chromosome[student, 0])
        
        new_harmony = TimeTableSolution(chromosome=new_chromosome)
        new_harmony.fitness = self.calculate_fitness(new_harmony)
        return new_harmony
