"""
Módulo que implementa el logging para el generador de datos sintéticos.

Autor: Juan José Jiménez González
Universidad: Universidad Isabel I
"""

import logging
import sys
from datetime import datetime

def setup_logging(output_dir):
    """
    Configura el sistema de logging para escribir en consola y archivo.
    
    Args:
        output_dir: Directorio donde crear el archivo de log
    """
    # Crear el logger
    logger = logging.getLogger('SyntheticDataGenerator')
    logger.setLevel(logging.INFO)
    
    # Formato para los mensajes
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # Handler para archivo
    file_handler = logging.FileHandler(f'{output_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Añadir handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TeeLogger:
    """
    Clase para capturar la salida estándar y escribirla tanto en consola como en archivo.
    """
    def __init__(self, filename):
        self.file = open(filename, 'a')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()