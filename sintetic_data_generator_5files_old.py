"""
Generador de datos sintéticos para el problema de asignación de tribunales.
Versión modificada para generación por lotes de 5 archivos.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from logging_implementation import setup_logging, TeeLogger
from typing import List, Dict, Any, Tuple

class SyntheticDataGenerator:
    def __init__(self):
        """Inicializa el generador de datos sintéticos."""
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.base_dir = "datos_sinteticos"
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.morning_slots = [
            f"{h:02d}:{m:02d}" for h in range(9, 14) 
            for m in (0, 30) if not (h == 13 and m == 30)
        ]
        self.afternoon_slots = [
            f"{h:02d}:{m:02d}" for h in range(16, 20) 
            for m in (0, 30) if not (h == 19 and m == 30)
        ]
        
        self.current_logger = None
        self.current_tee = None
        self.max_retry_attempts = 5  # Máximo número de intentos por escenario

    def _setup_batch_logging(self, batch_dir: str):
        """
        Configura el logging para un lote específico.
        
        Args:
            batch_dir: Directorio donde se guardará el log del lote
        """
        if self.current_logger:
            for handler in self.current_logger.handlers[:]:
                self.current_logger.removeHandler(handler)
        
        if self.current_tee:
            del self.current_tee
        
        self.current_logger = setup_logging(batch_dir)
        self.current_tee = TeeLogger(f'{batch_dir}/log.txt')
        
        self.current_logger.info(f"Iniciando generación de datos sintéticos en {batch_dir}")

    def _get_batch_directory(self, batch_num: int) -> str:
        """
        Obtiene el directorio para un lote específico.
        
        Args:
            batch_num: Número del lote actual
            
        Returns:
            str: Ruta al directorio del lote
        """
        start_idx = (batch_num * 5) + 1
        end_idx = start_idx + 4
        batch_dir = os.path.join(
            self.base_dir, 
            f"{self.timestamp}-{start_idx}-{end_idx}"
        )
        os.makedirs(batch_dir, exist_ok=True)
        return batch_dir

    def _calculate_slots_per_day(self):
        """Calcula el número de slots por día."""
        return len(self.morning_slots) + len(self.afternoon_slots)

    def _get_weekend_dates(self, num_slots_needed):
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

    def _matrix_to_dataframe(self, matrix: np.ndarray, row_prefix: str, 
                           col_names: List[str], first_col_name: str) -> pd.DataFrame:
        """
        Convierte una matriz de numpy a DataFrame con el formato requerido.
        
        Args:
            matrix: Matriz de datos
            row_prefix: Prefijo para los nombres de las filas
            col_names: Nombres de las columnas
            first_col_name: Nombre de la primera columna
            
        Returns:
            pd.DataFrame: DataFrame formateado
        """
        df = pd.DataFrame(matrix)
        df[df == 0] = np.nan
        
        df.insert(0, first_col_name, [f'{row_prefix}_{i+1}' for i in range(len(df))])
        df.columns = [first_col_name] + col_names
        
        return df
    
    def verify_feasibility(self, student_slots: np.ndarray, tribunal_slots: np.ndarray, 
                          compatibility: np.ndarray) -> bool:
        """
        Verifica que el escenario generado tenga al menos una solución válida.
        
        Args:
            student_slots: Matriz de disponibilidad estudiante-slots
            tribunal_slots: Matriz de disponibilidad tribunal-slots
            compatibility: Matriz de compatibilidad estudiante-tribunal
            
        Returns:
            bool: True si el escenario es factible, False en caso contrario
        """
        num_students = student_slots.shape[0]
        num_professors = tribunal_slots.shape[0]
        
        try:
            # 1. Verificar disponibilidad mínima
            for student in range(num_students):
                available_slots = np.sum(student_slots[student])
                if available_slots < 1:
                    return False
                
                compatible_profs = np.sum(compatibility[student])
                if compatible_profs < 3:
                    return False
            
            # 2. Verificar disponibilidad de profesores
            for prof in range(num_professors):
                if np.sum(tribunal_slots[prof]) < 3:
                    return False
            
            # 3. Verificar asignaciones viables
            for student in range(num_students):
                found_valid_assignment = False
                available_slots = np.where(student_slots[student] == 1)[0]
                compatible_profs = np.where(compatibility[student] == 1)[0]
                
                for slot in available_slots:
                    available_profs = [p for p in compatible_profs if tribunal_slots[p, slot] == 1]
                    if len(available_profs) >= 3:
                        found_valid_assignment = True
                        break
                
                if not found_valid_assignment:
                    return False
            
            return True
            
        except Exception as e:
            self.current_logger.error(f"Error en verificación de factibilidad: {str(e)}")
            return False

    def generate_scenario(self, num_students: int, num_professors: int, num_buildings: int = 1, 
                         num_aulas: int = 1, availability_rate: float = 0.3, 
                         compatibility_rate: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Genera un escenario sintético factible con disponibilidad realista.
        
        Args:
            num_students: Número de estudiantes
            num_professors: Número de profesores
            num_buildings: Número de edificios
            num_aulas: Número de aulas por edificio
            availability_rate: Tasa de disponibilidad
            compatibility_rate: Tasa de compatibilidad
            
        Returns:
            Tuple con matrices de disponibilidad y número total de slots
        """
        slots_per_day = self._calculate_slots_per_day()
        total_slots = num_buildings * num_aulas * slots_per_day
        
        if num_professors < 3:
            raise ValueError("Se necesitan al menos 3 profesores")
        if total_slots < num_students:
            raise ValueError("No hay suficientes slots para todos los estudiantes")
        
        # Aumentar tasas mínimas para garantizar factibilidad
        min_availability = max(0.4, availability_rate)
        min_compatibility = max(0.3, compatibility_rate)
        
        student_slots = np.zeros((num_students, total_slots))
        tribunal_slots = np.zeros((num_professors, total_slots))
        compatibility = np.zeros((num_students, num_professors))
        
        # 1. Crear solución base garantizada
        used_slots = set()
        assigned_tribunals = set()
        
        for student in range(num_students):
            # Asignar slot principal
            available_slots = list(set(range(total_slots)) - used_slots)
            if not available_slots:
                raise ValueError(f"No hay suficientes slots disponibles para el estudiante {student + 1}")
            
            main_slot = np.random.choice(available_slots)
            used_slots.add(main_slot)
            student_slots[student, main_slot] = 1
            
            # Asignar tribunal inicial
            available_profs = list(range(num_professors))
            tribunal = np.random.choice(available_profs, size=3, replace=False)
            
            for prof in tribunal:
                tribunal_slots[prof, main_slot] = 1
                compatibility[student, prof] = 1
                assigned_tribunals.add((main_slot, prof))
        
        # 2. Añadir disponibilidad adicional
        for student in range(num_students):
            num_extra_slots = max(3, int(total_slots * min_availability))
            available_extra = [s for s in range(total_slots) if student_slots[student, s] == 0]
            if available_extra:
                extra_slots = np.random.choice(
                    available_extra,
                    size=min(num_extra_slots, len(available_extra)),
                    replace=False
                )
                student_slots[student, extra_slots] = 1
        
        for prof in range(num_professors):
            num_extra_slots = max(4, int(total_slots * min_availability))
            available_extra = [s for s in range(total_slots) if tribunal_slots[prof, s] == 0]
            if available_extra:
                extra_slots = np.random.choice(
                    available_extra,
                    size=min(num_extra_slots, len(available_extra)),
                    replace=False
                )
                tribunal_slots[prof, extra_slots] = 1
        
        # 3. Añadir compatibilidades adicionales
        for student in range(num_students):
            num_compatible = max(4, int(num_professors * min_compatibility))
            current_compatible = set(np.where(compatibility[student] == 1)[0])
            available_profs = list(set(range(num_professors)) - current_compatible)
            
            if available_profs and len(current_compatible) < num_compatible:
                extra_profs = np.random.choice(
                    available_profs,
                    size=min(num_compatible - len(current_compatible), len(available_profs)),
                    replace=False
                )
                compatibility[student, extra_profs] = 1
        
        # Verificación final
        if not self.verify_feasibility(student_slots, tribunal_slots, compatibility):
            raise ValueError("No se pudo generar un escenario factible")
        
        return student_slots, tribunal_slots, compatibility, total_slots
    
    def create_excel(self, scenario_num: int, batch_dir: str, **kwargs) -> str:
        """
        Crea un archivo Excel con los datos generados.
        
        Args:
            scenario_num: Número del escenario actual
            batch_dir: Directorio donde guardar el archivo
            **kwargs: Parámetros para la generación del escenario
            
        Returns:
            str: Ruta al archivo Excel generado
        """
        try:
            student_slots, tribunal_slots, compatibility, total_slots = self.generate_scenario(**kwargs)
            
            filename = f"DatosGestionTribunales-{scenario_num:03d}.xlsx"
            filepath = os.path.join(batch_dir, filename)
            
            turnos_data = self._create_turnos_data(
                total_slots,
                kwargs.get('num_aulas', 1),
                kwargs.get('num_buildings', 1)
            )
            
            with pd.ExcelWriter(filepath) as writer:
                # Hoja de INFO
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
                
                df_student_slots.to_excel(writer, sheet_name='Disponibilidad-alumnos-turnos', index=False)
                df_tribunal_slots.to_excel(writer, sheet_name='Disponibilidad-tribunal-turnos', index=False)
                df_compatibility.to_excel(writer, sheet_name='Disponibilidad-alumnos-tribunal', index=False)
            
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"Error creando archivo Excel: {str(e)}")

    def generate_dataset_collection(self):
        """
        Genera una colección de conjuntos de datos organizados en lotes de 5 archivos.
        Cada lote se almacena en un directorio separado con su propio archivo de log.
        """
        scenarios = []
        slots_per_day = self._calculate_slots_per_day()
        generated_files = []
        
        # Generar configuraciones de escenarios
        for i in range(100):
            num_students = max(5, min(200, 5 + i * 2))
            num_professors = max(5, min(60, num_students // 2))
            min_slots_needed = num_students
            
            total_slots_per_room = slots_per_day
            rooms_needed = -(-min_slots_needed // total_slots_per_room)
            
            num_buildings = max(1, min(3, i // 35 + 1))
            num_aulas = max(1, -(-rooms_needed // num_buildings))
            
            total_slots = num_buildings * num_aulas * slots_per_day
            if total_slots < min_slots_needed:
                num_aulas = -(-min_slots_needed // (num_buildings * slots_per_day))
            
            availability_rate = max(0.4, min(0.8, 0.4 + (i / 200)))
            compatibility_rate = max(0.3, min(0.6, 0.3 + (i / 200)))
            
            scenarios.append({
                'num_students': num_students,
                'num_professors': num_professors,
                'num_buildings': num_buildings,
                'num_aulas': num_aulas,
                'availability_rate': availability_rate,
                'compatibility_rate': compatibility_rate
            })
        
        # Procesar escenarios en lotes de 5
        for batch_num in range(20):
            batch_dir = self._get_batch_directory(batch_num)
            self._setup_batch_logging(batch_dir)
            
            self.current_logger.info(f"\nProcesando lote {batch_num + 1}/20")
            start_idx = batch_num * 5
            
            # Asegurar que se generen exactamente 5 archivos por lote
            batch_files = []
            remaining_attempts = self.max_retry_attempts
            
            while len(batch_files) < 5 and remaining_attempts > 0:
                for i in range(5 - len(batch_files)):
                    scenario_idx = start_idx + len(batch_files)
                    scenario = scenarios[scenario_idx]
                    
                    try:
                        self.current_logger.info(f"\nGenerando escenario {scenario_idx + 1}:")
                        for key, value in scenario.items():
                            self.current_logger.info(f"{key}: {value}")
                        
                        filepath = self.create_excel(
                            scenario_idx + 1,
                            batch_dir,
                            **scenario
                        )
                        
                        batch_files.append(filepath)
                        self.current_logger.info(
                            f"Generado archivo {len(batch_files)}/5 del lote actual: "
                            f"{os.path.basename(filepath)}"
                        )
                        
                    except Exception as e:
                        self.current_logger.error(
                            f"Error en escenario {scenario_idx + 1}: {str(e)}")
                        self.current_logger.error("Detalles del escenario:")
                        for key, value in scenario.items():
                            self.current_logger.error(f"{key}: {value}")
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
        
        print(f"\nGeneración de datos completada")
        print(f"Total de archivos generados: {len(generated_files)}")
        print(f"Los archivos se han organizado en 20 carpetas en: {self.base_dir}")
        
        return generated_files

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    files = generator.generate_dataset_collection()