"""
Módulo principal con interfaz gráfica para la asignación optimizada de tribunales y horarios de TFG/TFM

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import os
import sys
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import matplotlib.pyplot as plt
from genetic_algorithm import TimetablingGA
from harmony_search import TimetablingHS
from visualization import (
    generate_professional_plots,
    generate_additional_plots
)

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Optimización de Tribunales - Universidad Isabel I")
        
        # Configurar ventana para resolución Full HD
        screen_width = 1920
        screen_height = 1080
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Marco principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Sección de selección de archivo
        file_frame = ttk.LabelFrame(main_frame, text="Selección de Archivo", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.file_path = tk.StringVar(value="DatosGestionTribunales.xlsx")
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=80)
        file_entry.grid(row=0, column=0, padx=5)
        
        browse_button = ttk.Button(file_frame, text="Examinar", command=self.browse_file)
        browse_button.grid(row=0, column=1, padx=5)
        
        # Botón de ejecución
        self.execute_button = ttk.Button(main_frame, text="Aplicar Algoritmos", command=self.execute_algorithms)
        self.execute_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Área de texto para la salida
        output_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        output_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.text_area = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=100, height=30)
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar redirección de salida
        sys.stdout = RedirectText(self.text_area)
        sys.stderr = RedirectText(self.text_area)
        
        # Indicador de progreso
        self.progress_frame = ttk.LabelFrame(main_frame, text="Progreso", padding="10")
        self.progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.grid(row=0, column=0)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=400, mode='indeterminate')
        self.progress_bar.grid(row=0, column=1, padx=10)
        
        # Lista de archivos generados
        output_files_frame = ttk.LabelFrame(main_frame, text="Soluciones Generadas", padding="10")
        output_files_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.output_files_listbox = tk.Listbox(output_files_frame, width=100, height=10)
        self.output_files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.output_files_listbox.bind('<Double-Button-1>', self.open_output_file)
        
        # Lista para almacenar los archivos de salida generados
        self.output_files = []

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def execute_algorithms(self):
        self.text_area.delete(1.0, tk.END)
        self.output_files_listbox.delete(0, tk.END)
        self.output_files = []
        
        self.execute_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Ejecutando algoritmos...")
        self.progress_bar.start(10)
        
        self.root.after(100, self.run_algorithms)

    def run_algorithms(self):
        print("="*80)
        print("SISTEMA DE OPTIMIZACIÓN PARA LA ASIGNACIÓN DE TRIBUNALES Y HORARIOS DE TFG/TFM")
        print("="*80)
        print("\nUniversidad Isabel I")
        print("-"*80)
        
        input_file = self.file_path.get()
        
        if not os.path.exists(input_file):
            print(f"\nError: El archivo '{input_file}' no existe en el directorio actual.")
            self.execute_button.config(state=tk.NORMAL)
            self.progress_bar.stop()
            self.progress_label.config(text="")
            return
            
        logo_path = 'logoui1.png'
        if not os.path.exists(logo_path):
            print(f"\nAdvertencia: No se encuentra el archivo del logo ({logo_path})")
            print("Las gráficas se generarán sin marca de agua.")
        
        results_dir = "resultados_comparacion"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            print("\nEjecutando Algoritmo Genético...")
            print("-"*50)
            ga_results = self.run_algorithm(TimetablingGA, input_file)
            
            print("\nEjecutando Harmony Search...")
            print("-"*50)
            hs_results = self.run_algorithm(TimetablingHS, input_file)
            
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            ga_output = os.path.join(results_dir, f"solucionAG-{base_filename}-{timestamp}.xlsx")
            hs_output = os.path.join(results_dir, f"solucionHS-{base_filename}-{timestamp}.xlsx")
            
            print("\nExportando soluciones...")
            print("-"*50)
            
            ga_algorithm = TimetablingGA(input_file)
            ga_algorithm.export_solution(ga_results[0], ga_output)
            
            hs_algorithm = TimetablingHS(input_file)
            hs_algorithm.export_solution(hs_results[0], hs_output)
            
            print("\nGenerando análisis comparativo con gráficas profesionales...")
            print("-"*50)
            
            generate_professional_plots(ga_results, hs_results, results_dir,
                                     f"{base_filename}_{timestamp}", logo_path)
            generate_additional_plots(ga_results, hs_results, results_dir,
                                   f"{base_filename}_{timestamp}", logo_path)
            
            print("\nProceso completado exitosamente.")
            print(f"\nLos resultados se han guardado en el directorio: {results_dir}")
            
            self.output_files.append(ga_output)
            self.output_files.append(hs_output)
            self.update_output_files_list()
            
        except Exception as e:
            print(f"\nError durante la ejecución: {str(e)}")
        
        finally:
            plt.close('all')
            self.execute_button.config(state=tk.NORMAL)
            self.progress_bar.stop()
            self.progress_label.config(text="")

    def run_algorithm(self, algorithm_class, input_file):
        import time
        start_time = time.time()
        algorithm = algorithm_class(input_file)
        best_solution, fitness_history = algorithm.solve()
        end_time = time.time()
        total_time = end_time - start_time
        generations = len(fitness_history)
        
        return best_solution, fitness_history, total_time, generations, start_time

    def update_output_files_list(self):
        self.output_files_listbox.delete(0, tk.END)
        for file_path in self.output_files:
            self.output_files_listbox.insert(tk.END, file_path)

    def open_output_file(self, event):
        selection = self.output_files_listbox.curselection()
        if selection:
            file_path = self.output_files_listbox.get(selection[0])
            os.startfile(file_path)

def main():
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()