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
        """
        Genera fechas de fin de semana en junio de 2024.
        """
        slots_per_day = self._calculate_slots_per_day()
        days_needed = -(-num_slots_needed // slots_per_day)  # Redondeo hacia arriba
        
        dates = []
        current_date = datetime(2024, 6, 1)  # Comenzar en junio 2024
        
        while len(dates) < days_needed:
            if current_date.weekday() == 4:  # Viernes
                dates.append(current_date)
            elif current_date.weekday() in [5, 6]:  # Sábado y domingo
                dates.append(current_date)
            current_date += timedelta(days=1)
            
            if current_date.month > 6:  # Si pasamos de junio, volver al inicio
                current_date = datetime(2024, 6, 1)
        
        return dates[:days_needed]

    def _create_turnos_data(self, num_slots, num_aulas, num_edificios):
        """
        Crea los datos de turnos asegurando consistencia en las dimensiones.
        """
        turnos_data = []
        slots_per_day = self._calculate_slots_per_day()
        total_slots_needed = num_slots
        dates = self._get_weekend_dates(total_slots_needed)
        
        turno_counter = 1
        for edificio in range(1, num_edificios + 1):
            for aula in range(1, num_aulas + 1):
                for date in dates:
                    # Slots de mañana
                    for hora in self.morning_slots:
                        if len(turnos_data) < total_slots_needed:
                            turnos_data.append({
                                'Turno': f"T{turno_counter}-A{aula}-E{edificio}",
                                'Fecha': date.strftime('%Y-%m-%d'),
                                'Hora': hora,
                                'Descripción': 'T=Turno, A=Aula, E=Edificio'
                            })
                            turno_counter += 1
                    
                    # Slots de tarde
                    for hora in self.afternoon_slots:
                        if len(turnos_data) < total_slots_needed:
                            turnos_data.append({
                                'Turno': f"T{turno_counter}-A{aula}-E{edificio}",
                                'Fecha': date.strftime('%Y-%m-%d'),
                                'Hora': hora,
                                'Descripción': 'T=Turno, A=Aula, E=Edificio'
                            })
                            turno_counter += 1
        
        return turnos_data

    def generate_scenario(self, num_students, num_professors, num_buildings=1, 
                         num_aulas=1, availability_rate=0.7, compatibility_rate=0.4):
        """
        Genera un escenario sintético factible.
        """
        slots_per_day = self._calculate_slots_per_day()
        total_slots = num_buildings * num_aulas * slots_per_day
        
        # Verificar factibilidad básica
        if num_professors < 3:
            raise ValueError("Se necesitan al menos 3 profesores")
        if total_slots < num_students:
            raise ValueError("No hay suficientes slots para todos los estudiantes")
        
        # Generar matrices de disponibilidad
        student_slots = np.full((num_students, total_slots), np.nan)
        tribunal_slots = np.full((num_professors, total_slots), np.nan)
        compatibility = np.full((num_students, num_professors), np.nan)
        
        # Asegurar una solución factible
        used_slots = set()
        for student in range(num_students):
            # Asignar slots al estudiante
            available_slots = list(set(range(total_slots)) - used_slots)
            if not available_slots:
                raise ValueError("No hay suficientes slots disponibles")
            
            main_slot = np.random.choice(available_slots)
            used_slots.add(main_slot)
            student_slots[student, main_slot] = 1
            
            # Asignar más slots según availability_rate
            extra_slots = np.random.choice(
                [s for s in range(total_slots) if s != main_slot],
                size=min(int(total_slots * availability_rate), total_slots - 1),
                replace=False
            )
            student_slots[student, extra_slots] = 1
            
            # Asignar profesores compatibles
            compatible_profs = np.random.choice(
                range(num_professors),
                size=max(3, int(num_professors * compatibility_rate)),
                replace=False
            )
            compatibility[student, compatible_profs] = 1
            
            # Asegurar disponibilidad de profesores
            for prof in compatible_profs:
                tribunal_slots[prof, main_slot] = 1
                # Añadir más slots según availability_rate
                extra_slots = np.random.choice(
                    range(total_slots),
                    size=int(total_slots * availability_rate),
                    replace=False
                )
                tribunal_slots[prof, extra_slots] = 1
        
        return student_slots, tribunal_slots, compatibility, total_slots

    def create_excel(self, scenario_num, **kwargs):
        """Crea un archivo Excel con los datos generados."""
        student_slots, tribunal_slots, compatibility, total_slots = self.generate_scenario(**kwargs)
        
        filename = f"DatosGestionTribunales-{scenario_num:03d}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        # Generar datos de turnos
        turnos_data = self._create_turnos_data(
            total_slots,
            kwargs.get('num_aulas', 1),
            kwargs.get('num_buildings', 1)
        )
        
        with pd.ExcelWriter(filepath) as writer:
            # Guardar hoja de Turnos
            pd.DataFrame(turnos_data).to_excel(writer, sheet_name='Turnos', index=False)
            
            # Guardar disponibilidad alumnos-turnos
            df_student_slots = pd.DataFrame(student_slots)
            df_student_slots.insert(0, 'Alumno', [f'Alumno_{i+1}' for i in range(student_slots.shape[0])])
            df_student_slots.columns = ['Alumno'] + [row['Turno'] for row in turnos_data]
            df_student_slots.to_excel(writer, sheet_name='Disponibilidad-alumnos-turnos', index=False)
            
            # Guardar disponibilidad tribunal-turnos
            df_tribunal_slots = pd.DataFrame(tribunal_slots)
            df_tribunal_slots.insert(0, 'Profesor', [f'Profesor_{i+1}' for i in range(tribunal_slots.shape[0])])
            df_tribunal_slots.columns = ['Profesor'] + [row['Turno'] for row in turnos_data]
            df_tribunal_slots.to_excel(writer, sheet_name='Disponibilidad-tribunal-turnos', index=False)
            
            # Guardar compatibilidad alumnos-tribunal
            df_compatibility = pd.DataFrame(compatibility)
            df_compatibility.insert(0, 'Alumno', [f'Alumno_{i+1}' for i in range(compatibility.shape[0])])
            df_compatibility.columns = ['Alumno'] + [f'Profesor_{i+1}' for i in range(compatibility.shape[1])]
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