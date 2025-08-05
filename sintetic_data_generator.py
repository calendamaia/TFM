"""
Generador de datos sintéticos para el problema de asignación de tribunales.
Versión corregida con mejor manejo de dimensiones y factibilidad.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from logging_implementation import setup_logging, TeeLogger

class SyntheticDataGenerator:
    def __init__(self):
        """Inicializa el generador de datos sintéticos."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.base_dir = "datos_sinteticos"
        self.output_dir = os.path.join(self.base_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar logging
        self.logger = setup_logging(self.output_dir)
        # Capturar toda la salida estándar
        self.tee = TeeLogger(f'{self.output_dir}/log.txt')
        
        self.logger.info(f"Iniciando generación de datos sintéticos en {self.output_dir}")

        self.morning_slots = [
            f"{h:02d}:{m:02d}" for h in range(9, 14) 
            for m in (0, 30) if not (h == 13 and m == 30)
        ]
        self.afternoon_slots = [
            f"{h:02d}:{m:02d}" for h in range(16, 20) 
            for m in (0, 30) if not (h == 19 and m == 30)
        ]

    def _calculate_slots_per_day(self):
        """Calcula el número de slots por día."""
        return len(self.morning_slots) + len(self.afternoon_slots)

    def _get_weekend_dates(self, num_slots_needed):
        """Genera fechas de fin de semana en junio de 2024."""
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

    def _create_turnos_data(self, num_slots, num_aulas, num_edificios):
        """Crea los datos de turnos con dimensiones consistentes."""
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

    def _matrix_to_dataframe(self, matrix, row_prefix, col_names, first_col_name):
        """
        Convierte una matriz de numpy a DataFrame con el formato requerido.
        Los ceros se convierten en valores vacíos.
        """
        df = pd.DataFrame(matrix)
        df[df == 0] = np.nan  # Convertir ceros a NaN
        
        # Agregar columna de identificación
        df.insert(0, first_col_name, [f'{row_prefix}_{i+1}' for i in range(len(df))])
        
        # Renombrar columnas
        df.columns = [first_col_name] + col_names
        
        return df

    def generate_scenario(self, num_students, num_professors, num_buildings=1, 
                         num_aulas=1, availability_rate=0.3, compatibility_rate=0.3):
        """
        Genera un escenario sintético factible con disponibilidad realista.
        La generación  sigue un proceso en tres fases:

        - Primero crea una solución base garantizada
        - Luego añade disponibilidad adicional de forma controlada
        - Finalmente verifica y ajusta para asegurar mínimos razonables
       
        """
        slots_per_day = self._calculate_slots_per_day()
        total_slots = num_buildings * num_aulas * slots_per_day
        
        if num_professors < 3:
            raise ValueError("Se necesitan al menos 3 profesores")
        if total_slots < num_students:
            raise ValueError("No hay suficientes slots para todos los estudiantes")
        
        # Inicializar matrices
        student_slots = np.zeros((num_students, total_slots))
        tribunal_slots = np.zeros((num_professors, total_slots))
        compatibility = np.zeros((num_students, num_professors))
        
        # 1. Primero, crear una solución base garantizada
        used_slots = set()
        assigned_tribunals = set()  # Para seguir las asignaciones profesor-slot
        
        for student in range(num_students):
            # Encontrar un slot disponible
            available_slots = list(set(range(total_slots)) - used_slots)
            if not available_slots:
                raise ValueError("No hay suficientes slots disponibles")
            
            main_slot = np.random.choice(available_slots)
            used_slots.add(main_slot)
            
            # Seleccionar tres profesores que no estén ya asignados en ese slot
            available_profs = [p for p in range(num_professors) 
                             if (main_slot, p) not in assigned_tribunals]
            
            if len(available_profs) < 3:
                # Si no hay suficientes profesores disponibles, reorganizar asignaciones
                available_profs = list(range(num_professors))
            
            tribunal = np.random.choice(available_profs, size=3, replace=False)
            
            # Registrar las asignaciones base
            student_slots[student, main_slot] = 1
            for prof in tribunal:
                tribunal_slots[prof, main_slot] = 1
                compatibility[student, prof] = 1
                assigned_tribunals.add((main_slot, prof))
        
        # 2. Añadir disponibilidad adicional de forma controlada
        # Para estudiantes
        for student in range(num_students):
            num_extra_slots = max(2, int(total_slots * availability_rate))
            extra_slots = np.random.choice(
                [s for s in range(total_slots) if student_slots[student, s] == 0],
                size=min(num_extra_slots, total_slots - 1),
                replace=False
            )
            student_slots[student, extra_slots] = 1
        
        # Para profesores
        for prof in range(num_professors):
            # Calcular slots donde el profesor ya está asignado
            assigned_slots = np.where(tribunal_slots[prof] == 1)[0]
            
            # Añadir disponibilidad adicional
            num_extra_slots = max(3, int(total_slots * availability_rate))
            available_slots = [s for s in range(total_slots) 
                             if s not in assigned_slots]
            
            if available_slots:
                extra_slots = np.random.choice(
                    available_slots,
                    size=min(num_extra_slots, len(available_slots)),
                    replace=False
                )
                tribunal_slots[prof, extra_slots] = 1
        
        # Para compatibilidad alumno-tribunal
        for student in range(num_students):
            # Añadir algunos profesores compatibles adicionales
            current_compatible = set(np.where(compatibility[student] == 1)[0])
            available_profs = set(range(num_professors)) - current_compatible
            
            if available_profs:
                num_extra_profs = max(2, int(num_professors * compatibility_rate))
                extra_profs = np.random.choice(
                    list(available_profs),
                    size=min(num_extra_profs, len(available_profs)),
                    replace=False
                )
                compatibility[student, extra_profs] = 1
        
        # 3. Verificar y ajustar si es necesario
        for prof in range(num_professors):
            # Asegurar que cada profesor tiene al menos algunos slots
            if np.sum(tribunal_slots[prof]) < 3:
                available_slots = [s for s in range(total_slots) 
                                 if tribunal_slots[prof, s] == 0]
                if available_slots:
                    extra_slots = np.random.choice(
                        available_slots,
                        size=min(3, len(available_slots)),
                        replace=False
                    )
                    tribunal_slots[prof, extra_slots] = 1
            
            # Asegurar que cada profesor es compatible con algunos estudiantes
            if np.sum(compatibility[:, prof]) < 2:
                available_students = [s for s in range(num_students) 
                                   if compatibility[s, prof] == 0]
                if available_students:
                    extra_students = np.random.choice(
                        available_students,
                        size=min(2, len(available_students)),
                        replace=False
                    )
                    compatibility[extra_students, prof] = 1
        
        return student_slots, tribunal_slots, compatibility, total_slots

    def create_excel(self, scenario_num, **kwargs):
        """Crea un archivo Excel con los datos generados."""
        student_slots, tribunal_slots, compatibility, total_slots = self.generate_scenario(**kwargs)
        
        filename = f"DatosGestionTribunales-{scenario_num:03d}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        turnos_data = self._create_turnos_data(
            total_slots,
            kwargs.get('num_aulas', 1),
            kwargs.get('num_buildings', 1)
        )
        
        with pd.ExcelWriter(filepath) as writer:
            # Guardar hoja de Turnos
            pd.DataFrame(turnos_data).to_excel(writer, sheet_name='Turnos', index=False)
            
            # Convertir matrices a DataFrames con el formato requerido
            turno_names = [row['Turno'] for row in turnos_data]
            profesor_names = [f'Profesor_{i+1}' for i in range(tribunal_slots.shape[0])]
            
            # Disponibilidad alumnos-turnos
            df_student_slots = self._matrix_to_dataframe(
                student_slots, 'Alumno', turno_names, 'Alumno')
            
            # Disponibilidad tribunal-turnos
            df_tribunal_slots = self._matrix_to_dataframe(
                tribunal_slots, 'Profesor', turno_names, 'Profesor')
            
            # Compatibilidad alumnos-tribunal
            df_compatibility = self._matrix_to_dataframe(
                compatibility, 'Alumno', profesor_names, 'Alumno')
            
            # Guardar en Excel
            df_student_slots.to_excel(writer, sheet_name='Disponibilidad-alumnos-turnos', index=False)
            df_tribunal_slots.to_excel(writer, sheet_name='Disponibilidad-tribunal-turnos', index=False)
            df_compatibility.to_excel(writer, sheet_name='Disponibilidad-alumnos-tribunal', index=False)
        
        return filepath

    def generate_dataset_collection(self):
        """
        Genera una colección de conjuntos de datos con complejidad creciente,
        asegurando suficientes slots para todos los estudiantes.
        """
        scenarios = []
        slots_per_day = self._calculate_slots_per_day()
        self.logger.info("Comenzando generación de escenarios")
        
        for i in range(100):
            # Calcular estudiantes y profesores
            num_students = max(5, min(200, 5 + i * 2))
            num_professors = max(5, min(60, num_students // 2))
            
            # Calcular número mínimo de slots necesarios
            min_slots_needed = num_students  # Al menos un slot por estudiante
            
            # Calcular edificios y aulas necesarios
            total_slots_per_room = slots_per_day
            rooms_needed = -(-min_slots_needed // total_slots_per_room)  # Redondeo hacia arriba
            
            # Distribuir rooms_needed entre edificios y aulas
            num_buildings = max(1, min(3, i // 35 + 1))  # Máximo 3 edificios
            num_aulas = max(1, -(-rooms_needed // num_buildings))  # Redondeo hacia arriba
            
            # Verificar que tenemos suficientes slots
            total_slots = num_buildings * num_aulas * slots_per_day
            if total_slots < min_slots_needed:
                # Ajustar aulas si es necesario
                num_aulas = -(-min_slots_needed // (num_buildings * slots_per_day))
            
            # Ajustar tasas para mantener factibilidad
            availability_rate = max(0.4, min(0.8, 0.4 + (i / 200)))
            compatibility_rate = max(0.3, min(0.6, 0.3 + (i / 200)))

            # Logging de información del escenario
            self.logger.info(f"\nEscenario {i+1}:")
            self.logger.info(f"Estudiantes: {num_students}")
            self.logger.info(f"Profesores: {num_professors}")
            self.logger.info(f"Edificios: {num_buildings}")
            self.logger.info(f"Aulas por edificio: {num_aulas}")
            self.logger.info(f"Slots por día: {slots_per_day}")
            self.logger.info(f"Total slots disponibles: {total_slots}")
            self.logger.info(f"Slots mínimos necesarios: {min_slots_needed}")
            
            scenarios.append({
                'num_students': num_students,
                'num_professors': num_professors,
                'num_buildings': num_buildings,
                'num_aulas': num_aulas,
                'availability_rate': availability_rate,
                'compatibility_rate': compatibility_rate
            })
            
            # Imprimir información de debug
            print(f"\nEscenario {i+1}:")
            print(f"Estudiantes: {num_students}")
            print(f"Profesores: {num_professors}")
            print(f"Edificios: {num_buildings}")
            print(f"Aulas por edificio: {num_aulas}")
            print(f"Slots por día: {slots_per_day}")
            print(f"Total slots disponibles: {total_slots}")
            print(f"Slots mínimos necesarios: {min_slots_needed}")
        
        generated_files = []
        for i, scenario in enumerate(scenarios, 1):
            try:
                filepath = self.create_excel(i, **scenario)
                generated_files.append(filepath)
                self.logger.info(f"Generado archivo {i}/100: {os.path.basename(filepath)}")
            except Exception as e:
                self.logger.error(f"Error en escenario {i}: {str(e)}")
                self.logger.error(f"Detalles del escenario {i}:")
                for key, value in scenario.items():
                    self.logger.error(f"{key}: {value}")
                continue
        
        self.logger.info(f"\nArchivos generados en: {self.output_dir}")
        self.logger.info(f"Total de archivos generados: {len(generated_files)}")
        
        return generated_files

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    files = generator.generate_dataset_collection()
    print(f"\nArchivos generados en: {generator.output_dir}")
    print(f"Total de archivos generados: {len(files)}")