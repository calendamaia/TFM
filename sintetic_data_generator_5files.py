"""
Generador de datos sintéticos para el problema de asignación de tribunales.
Versión mejorada para garantizar la factibilidad de las soluciones.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
Fecha: 2024
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from logging_implementation import setup_logging, TeeLogger

class SyntheticDataGenerator:
    """Clase para la generación de datos sintéticos factibles."""
    
    def __init__(self):
        """Inicializa el generador con configuraciones base."""
        # Configuración de directorios
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.base_dir = "datos_sinteticos"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Definición de slots temporales
        self.morning_slots = [
            f"{h:02d}:{m:02d}" for h in range(9, 14) 
            for m in (0, 30) if not (h == 13 and m == 30)
        ]
        self.afternoon_slots = [
            f"{h:02d}:{m:02d}" for h in range(16, 20) 
            for m in (0, 30) if not (h == 19 and m == 30)
        ]
        
        # Configuración de generación
        self.max_retry_attempts = 5
        self.min_slots_per_student = 3
        self.min_slots_per_professor = 4
        self.min_compatible_professors = 4
        
        # Tasas mínimas para garantizar factibilidad
        self.min_availability_rate = 0.4
        self.min_compatibility_rate = 0.3
        
        # Parámetros para generación incremental
        self.student_ranges = [(5, 15), (16, 30), (31, 50), (51, 100), (101, 200)]
        self.building_ranges = [1, 1, 2, 2, 3]
        
        # Configuración de logging
        self.current_logger = None
        self.current_tee = None

    def _setup_batch_logging(self, batch_dir: str) -> None:
        """Configura el sistema de logging para un lote específico."""
        if self.current_logger:
            for handler in self.current_logger.handlers[:]:
                self.current_logger.removeHandler(handler)
        
        if self.current_tee:
            del self.current_tee
        
        self.current_logger = setup_logging(batch_dir)
        self.current_tee = TeeLogger(f'{batch_dir}/log.txt')
        
        self.current_logger.info(f"Iniciando generación de datos sintéticos en {batch_dir}")

    def _get_batch_directory(self, batch_num: int) -> str:
        """Obtiene el directorio para un lote específico."""
        start_idx = (batch_num * 5) + 1
        end_idx = start_idx + 4
        batch_dir = os.path.join(
            self.base_dir, 
            f"{self.timestamp}-{start_idx}-{end_idx}"
        )
        os.makedirs(batch_dir, exist_ok=True)
        return batch_dir

    def _calculate_slots_per_day(self) -> int:
        """Calcula el número total de slots temporales por día."""
        return len(self.morning_slots) + len(self.afternoon_slots)

    def _validate_dimensions(self, num_students: int, num_professors: int,
                          num_buildings: int, num_aulas: int) -> bool:
        """
        Valida que las dimensiones del problema permitan una solución.
        
        Args:
            num_students: Número de estudiantes
            num_professors: Número de profesores
            num_buildings: Número de edificios
            num_aulas: Número de aulas
            
        Returns:
            bool: True si las dimensiones son válidas
        """
        slots_per_day = self._calculate_slots_per_day()
        total_slots = num_buildings * num_aulas * slots_per_day
        
        # Validaciones básicas
        if num_professors < 3:
            self.current_logger.error("Se requieren al menos 3 profesores")
            return False
            
        if total_slots < num_students:
            self.current_logger.error(
                f"Slots insuficientes ({total_slots}) para {num_students} estudiantes"
            )
            return False
            
        # Validación de capacidad de tribunales
        max_tribunals_per_slot = num_professors // 3
        if max_tribunals_per_slot * total_slots < num_students:
            self.current_logger.error(
                f"Capacidad de tribunales insuficiente para {num_students} estudiantes"
            )
            return False
        
        return True
    def _generate_guaranteed_solution(self, num_students: int, num_professors: int,
                                    total_slots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera una solución base garantizada.
        
        Args:
            num_students: Número de estudiantes
            num_professors: Número de profesores
            total_slots: Número total de slots temporales
            
        Returns:
            Tuple con las matrices de disponibilidad y compatibilidad
        """
        student_slots = np.zeros((num_students, total_slots))
        tribunal_slots = np.zeros((num_professors, total_slots))
        compatibility = np.zeros((num_students, num_professors))
        
        # Lista de slots disponibles
        available_slots = list(range(total_slots))
        random.shuffle(available_slots)
        
        # Asignar slots y tribunales base
        for student in range(num_students):
            # Verificar slots disponibles
            if not available_slots:
                raise ValueError("No hay suficientes slots para asignar")
            
            # Asignar slot
            slot = available_slots.pop(0)
            student_slots[student, slot] = 1
            
            # Asignar tribunal garantizado
            available_profs = list(range(num_professors))
            random.shuffle(available_profs)
            tribunal = available_profs[:3]
            
            for prof in tribunal:
                tribunal_slots[prof, slot] = 1
                compatibility[student, prof] = 1
        
        return student_slots, tribunal_slots, compatibility

    def _add_extra_availability(self, student_slots: np.ndarray,
                              tribunal_slots: np.ndarray,
                              compatibility: np.ndarray,
                              availability_rate: float,
                              compatibility_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Añade disponibilidad adicional de forma controlada.
        
        Args:
            student_slots: Matriz de disponibilidad estudiante-slots
            tribunal_slots: Matriz de disponibilidad tribunal-slots
            compatibility: Matriz de compatibilidad estudiante-tribunal
            availability_rate: Tasa de disponibilidad deseada
            compatibility_rate: Tasa de compatibilidad deseada
            
        Returns:
            Tuple con las matrices actualizadas
        """
        num_students, total_slots = student_slots.shape
        num_professors = tribunal_slots.shape[0]
        
        # Añadir slots adicionales para estudiantes
        for student in range(num_students):
            current_slots = np.where(student_slots[student] == 0)[0]
            num_extra = max(
                self.min_slots_per_student,
                int(total_slots * availability_rate)
            )
            if len(current_slots) > 0:
                extra_slots = np.random.choice(
                    current_slots,
                    size=min(num_extra, len(current_slots)),
                    replace=False
                )
                student_slots[student, extra_slots] = 1
        
        # Añadir slots adicionales para profesores
        for prof in range(num_professors):
            current_slots = np.where(tribunal_slots[prof] == 0)[0]
            num_extra = max(
                self.min_slots_per_professor,
                int(total_slots * availability_rate)
            )
            if len(current_slots) > 0:
                extra_slots = np.random.choice(
                    current_slots,
                    size=min(num_extra, len(current_slots)),
                    replace=False
                )
                tribunal_slots[prof, extra_slots] = 1
        
        # Añadir compatibilidades adicionales
        for student in range(num_students):
            current_compatible = set(np.where(compatibility[student] == 1)[0])
            available_profs = list(set(range(num_professors)) - current_compatible)
            
            num_extra = max(
                self.min_compatible_professors - len(current_compatible),
                int(num_professors * compatibility_rate) - len(current_compatible)
            )
            
            if available_profs and num_extra > 0:
                extra_profs = np.random.choice(
                    available_profs,
                    size=min(num_extra, len(available_profs)),
                    replace=False
                )
                compatibility[student, extra_profs] = 1
        
        return student_slots, tribunal_slots, compatibility
    
    def _get_weekend_dates(self, num_slots_needed: int) -> List[datetime]:
        """
        Genera fechas de fin de semana en junio de 2024.
        
        Args:
            num_slots_needed: Número de slots temporales necesarios
            
        Returns:
            List[datetime]: Lista de fechas de fin de semana
        """
        slots_per_day = self._calculate_slots_per_day()
        days_needed = -(-num_slots_needed // slots_per_day)
        
        dates = []
        current_date = datetime(2024, 6, 1)
        
        while len(dates) < days_needed:
            if current_date.weekday() >= 4:  # Viernes, sábado o domingo
                dates.append(current_date)
            current_date += timedelta(days=1)
            if current_date.month > 6:
                current_date = datetime(2024, 6, 1)
        
        return dates[:days_needed]

    def _create_turnos_data(self, num_slots: int, num_aulas: int, num_edificios: int) -> List[Dict]:
        """
        Crea los datos de turnos con dimensiones consistentes.
        
        Args:
            num_slots: Número total de slots temporales
            num_aulas: Número de aulas por edificio
            num_edificios: Número de edificios
            
        Returns:
            List[Dict]: Lista de diccionarios con información de turnos
        """
        turnos_data = []
        slots_per_day = self._calculate_slots_per_day()
        dates = self._get_weekend_dates(num_slots)
        
        turno_counter = 1
        for edificio in range(1, num_edificios + 1):
            for aula in range(1, num_aulas + 1):
                for date in dates:
                    for hora in self.morning_slots + self.afternoon_slots:
                        if len(turnos_data) < num_slots:
                            turnos_data.append({
                                'Turno': f"T{turno_counter}-A{aula}-E{edificio}",
                                'Fecha': date.strftime('%Y-%m-%d'),
                                'Hora': hora,
                                'Descripción': 'T=Turno, A=Aula, E=Edificio'
                            })
                            turno_counter += 1
        
        return turnos_data

    def verify_feasibility(self, student_slots: np.ndarray,
                          tribunal_slots: np.ndarray,
                          compatibility: np.ndarray) -> bool:
        """
        Verifica que exista al menos una solución válida.
        
        Args:
            student_slots: Matriz de disponibilidad estudiante-slots
            tribunal_slots: Matriz de disponibilidad tribunal-slots
            compatibility: Matriz de compatibilidad estudiante-tribunal
            
        Returns:
            bool: True si existe al menos una solución válida
        """
        num_students = student_slots.shape[0]
        used_slots: Set[int] = set()
        
        for student in range(num_students):
            found_valid_slot = False
            
            # Buscar slots disponibles para el estudiante
            for slot in range(student_slots.shape[1]):
                if slot in used_slots or student_slots[student, slot] == 0:
                    continue
                
                # Verificar si hay suficientes profesores disponibles
                compatible_profs = np.where(compatibility[student] == 1)[0]
                available_profs = [p for p in compatible_profs 
                                 if tribunal_slots[p, slot] == 1]
                
                if len(available_profs) >= 3:
                    used_slots.add(slot)
                    found_valid_slot = True
                    break
            
            if not found_valid_slot:
                self.current_logger.warning(
                    f"No se encontró slot válido para estudiante {student}"
                )
                return False
        
        return True
    
    def generate_scenario(self, num_students: int, num_professors: int,
                         num_buildings: int = 1, num_aulas: int = 1,
                         availability_rate: float = 0.3,
                         compatibility_rate: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Genera un escenario factible con al menos una solución válida.
        
        Args:
            num_students: Número de estudiantes
            num_professors: Número de profesores
            num_buildings: Número de edificios
            num_aulas: Número de aulas por edificio
            availability_rate: Tasa de disponibilidad
            compatibility_rate: Tasa de compatibilidad
            
        Returns:
            Tuple con las matrices y el número total de slots
        """
        slots_per_day = self._calculate_slots_per_day()
        total_slots = num_buildings * num_aulas * slots_per_day
        
        # Validar dimensiones
        if not self._validate_dimensions(num_students, num_professors,
                                       num_buildings, num_aulas):
            raise ValueError("Dimensiones del problema no válidas")
        
        # Establecer tasas mínimas
        availability_rate = max(self.min_availability_rate, availability_rate)
        compatibility_rate = max(self.min_compatibility_rate, compatibility_rate)
        
        # Generar solución base garantizada
        student_slots, tribunal_slots, compatibility = self._generate_guaranteed_solution(
            num_students, num_professors, total_slots
        )
        
        # Añadir disponibilidad adicional
        student_slots, tribunal_slots, compatibility = self._add_extra_availability(
            student_slots, tribunal_slots, compatibility,
            availability_rate, compatibility_rate
        )
        
        # Verificar factibilidad
        if not self.verify_feasibility(student_slots, tribunal_slots, compatibility):
            raise ValueError("No se pudo generar un escenario factible")
        
        return student_slots, tribunal_slots, compatibility, total_slots

    def create_excel(self, scenario_num: int, batch_dir: str, **kwargs) -> str:
        """
        Crea el archivo Excel con los datos generados.
        
        Args:
            scenario_num: Número del escenario
            batch_dir: Directorio donde guardar el archivo
            **kwargs: Parámetros para la generación del escenario
            
        Returns:
            str: Ruta al archivo Excel generado
        """
        try:
            # Generar escenario factible
            student_slots, tribunal_slots, compatibility, total_slots = \
                self.generate_scenario(**kwargs)
            
            # Crear archivo Excel
            filename = f"DatosGestionTribunales-{scenario_num:03d}.xlsx"
            filepath = os.path.join(batch_dir, filename)
            
            # Generar datos de turnos
            turnos_data = self._create_turnos_data(
                total_slots,
                kwargs.get('num_aulas', 1),
                kwargs.get('num_buildings', 1)
            )
            
            # Crear Excel
            with pd.ExcelWriter(filepath) as writer:
                # Hoja INFO
                info_data = {
                    'Parámetro': [
                        'Número de estudiantes',
                        'Número de profesores',
                        'Número de edificios',
                        'Número de aulas',
                        'Tasa de disponibilidad',
                        'Tasa de compatibilidad'
                    ],
                    'Valor': [
                        kwargs['num_students'],
                        kwargs['num_professors'],
                        kwargs['num_buildings'],
                        kwargs['num_aulas'],
                        kwargs['availability_rate'],
                        kwargs['compatibility_rate']
                    ]
                }
                pd.DataFrame(info_data).to_excel(writer, sheet_name='INFO', index=False)

                # Hoja de Turnos
                pd.DataFrame(turnos_data).to_excel(writer, sheet_name='Turnos', index=False)

                # Preparar nombres de columnas
                turno_names = [row['Turno'] for row in turnos_data]
                profesor_names = [f'Profesor_{i+1}' for i in range(tribunal_slots.shape[0])]

                # Hojas de disponibilidad y compatibilidad
                df_student_slots = self._matrix_to_dataframe(
                    student_slots, 'Alumno', turno_names, 'Alumno')
                df_tribunal_slots = self._matrix_to_dataframe(
                    tribunal_slots, 'Profesor', turno_names, 'Profesor')
                df_compatibility = self._matrix_to_dataframe(
                    compatibility, 'Alumno', profesor_names, 'Alumno')

                df_student_slots.to_excel(writer, sheet_name='Disponibilidad-alumnos-turnos',
                                      index=False)
                df_tribunal_slots.to_excel(writer, sheet_name='Disponibilidad-tribunal-turnos',
                                       index=False)
                df_compatibility.to_excel(writer, sheet_name='Disponibilidad-alumnos-tribunal',
                                      index=False)

            return filepath

        except Exception as e:
            raise RuntimeError(f"Error creando archivo Excel: {str(e)}")

    def generate_dataset_collection(self) -> List[str]:
        """
        Genera una colección de datos en lotes de 5 archivos.
        
        Returns:
            List[str]: Lista de rutas a los archivos generados
        """
        # Generar configuraciones base
        base_configs = []
        slots_per_day = self._calculate_slots_per_day()

        for i in range(100):
            # Calcular dimensiones incrementales
            num_students = max(5, min(200, 5 + i * 2))
            num_professors = max(5, min(60, num_students // 2))
            
            # Calcular edificios y aulas necesarios
            min_slots_needed = num_students
            num_buildings = max(1, min(3, i // 35 + 1))
            rooms_needed = -(-min_slots_needed // (slots_per_day * num_buildings))
            num_aulas = max(1, rooms_needed)
            
            # Calcular tasas adaptativas
            progress_factor = i / 100
            availability_rate = self.min_availability_rate + (0.4 * progress_factor)
            compatibility_rate = self.min_compatibility_rate + (0.3 * progress_factor)
            
            base_configs.append({
                'num_students': num_students,
                'num_professors': num_professors,
                'num_buildings': num_buildings,
                'num_aulas': num_aulas,
                'availability_rate': availability_rate,
                'compatibility_rate': compatibility_rate
            })

        generated_files = []
        
        # Generar lotes de 5 archivos
        for batch_num in range(20):
            batch_dir = self._get_batch_directory(batch_num)
            self._setup_batch_logging(batch_dir)
            
            self.current_logger.info(f"\nProcesando lote {batch_num + 1}/20")
            start_idx = batch_num * 5
            
            # Generar exactamente 5 archivos por lote
            batch_files = []
            remaining_attempts = self.max_retry_attempts
            
            while len(batch_files) < 5 and remaining_attempts > 0:
                for i in range(5 - len(batch_files)):
                    scenario_idx = start_idx + len(batch_files)
                    config = base_configs[scenario_idx]
                    
                    try:
                        self.current_logger.info(f"\nGenerando escenario {scenario_idx + 1}:")
                        for key, value in config.items():
                            self.current_logger.info(f"{key}: {value}")
                        
                        filepath = self.create_excel(
                            scenario_idx + 1,
                            batch_dir,
                            **config
                        )
                        
                        batch_files.append(filepath)
                        self.current_logger.info(
                            f"Archivo {len(batch_files)}/5 generado: "
                            f"{os.path.basename(filepath)}"
                        )
                        
                    except Exception as e:
                        self.current_logger.error(
                            f"Error en escenario {scenario_idx + 1}: {str(e)}"
                        )
                        continue
                
                if len(batch_files) < 5:
                    remaining_attempts -= 1
                    self.current_logger.warning(
                        f"Lote incompleto. Reintentando... "
                        f"(intentos restantes: {remaining_attempts})"
                    )
            
            if len(batch_files) < 5:
                raise RuntimeError(
                    f"No se pudieron generar 5 archivos en el lote {batch_num + 1} "
                    f"después de {self.max_retry_attempts} intentos"
                )
            
            generated_files.extend(batch_files)
            self.current_logger.info(f"\nLote {batch_num + 1} completado")
            self.current_logger.info(f"Archivos guardados en: {batch_dir}")
        
        self.current_logger.info(f"\nGeneración completada")
        self.current_logger.info(f"Total archivos generados: {len(generated_files)}")
        
        return generated_files
    
    def _matrix_to_dataframe(self, matrix: np.ndarray, row_prefix: str,
                           col_names: List[str], first_col_name: str) -> pd.DataFrame:
        """
        Convierte una matriz de numpy a DataFrame con el formato requerido.
        Los ceros se convierten en valores vacíos.
        
        Args:
            matrix: Matriz de datos
            row_prefix: Prefijo para los nombres de las filas
            col_names: Nombres de las columnas
            first_col_name: Nombre de la primera columna
            
        Returns:
            pd.DataFrame: DataFrame formateado
        """
        # Convertir la matriz a DataFrame
        df = pd.DataFrame(matrix)
        
        # Convertir ceros a NaN (valores vacíos)
        df[df == 0] = np.nan
        
        # Agregar columna de identificación
        df.insert(0, first_col_name, [f'{row_prefix}_{i+1}' for i in range(len(df))])
        
        # Renombrar columnas
        df.columns = [first_col_name] + col_names
        
        return df

def main():
    """Función principal del programa."""
    try:
        generator = SyntheticDataGenerator()
        files = generator.generate_dataset_collection()
        print(f"\nGeneración completada exitosamente")
        print(f"Total archivos generados: {len(files)}")
        
    except Exception as e:
        print(f"Error durante la generación: {str(e)}")
        raise

if __name__ == "__main__":
    main()