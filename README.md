# Sistema de Optimización de Tribunales TFM

**Autor:** Juan José Jiménez González  
**Universidad:** Universidad Isabel I  
**Máster:** Análisis Inteligente de Datos (Big Data)


## Descripción

Sistema de optimización para la asignación de tribunales de Trabajo Fin de Máster (TFM) mediante algoritmos evolutivos. El sistema compara el rendimiento de dos algoritmos de optimización: Algoritmo Genético (GA) y Harmony Search (HS) aplicados al problema multicriterio de asignación de tribunales académicos.

## Instrucciones de Uso

Para utilizar el sistema, es necesario ejecutar los ficheros en el siguiente orden:

### 1. Generación de Datos Sintéticos
```bash
python sintetic_data_generator_5files.py
```
Este script genera conjuntos de datos sintéticos de prueba organizados en 100 ficheros Excel con diferentes niveles de complejidad. Los datos se almacenan en la carpeta `datos_sinteticos` con estructura temporal y numeración secuencial.

### 2. Ejecución de Algoritmos de Optimización
```bash
python batch_main.py
```
Módulo principal que ejecuta ambos algoritmos (GA y HS) sobre los datos generados. Procesa cada fichero de entrada, ejecuta los algoritmos de optimización, genera las soluciones y crea gráficas comparativas individuales. Los resultados se almacenan en subcarpetas con formato timestamp.

### 3. Análisis Comparativo Global
```bash
python analysis_visualizer_5files.py
```
Realiza el análisis estadístico comparativo del conjunto completo de 100 casos. Genera reportes consolidados, visualizaciones comparativas globales y métricas de rendimiento. Los resultados se guardan en la subcarpeta `análisis_complete`.

### 4. Visualización de Resultados Específicos (Opcional)
```bash
python sintetic_visualization.py
```
Script complementario para analizar y visualizar resultados de ejecuciones específicas, permitiendo seleccionar carpetas de datos para análisis detallado.

## Datos de Prueba Incluidos

El repositorio incluye datos de prueba ya generados en la carpeta `datos_sinteticos/20241225-125217-1-5` que contiene 5 ficheros Excel de ejemplo (`DatosGestionTribunales-001.xlsx` a `DatosGestionTribunales-005.xlsx`). Estos datos permiten ejecutar directamente los algoritmos de optimización sin necesidad de generar nuevos datos sintéticos.

## Estructura de Datos

- **Entrada:** Archivos Excel con formato `DatosGestionTribunales-xxx.xlsx` (donde xxx va de 001 a 100)
- **Salida:** Soluciones optimizadas, gráficas comparativas y reportes estadísticos en formato Excel y PNG

## Requisitos del Sistema

- Python 3.x
- Librerías: pandas, numpy, matplotlib, openpyxl
- Estructura de directorios del proyecto TFM

## Notas Importantes

- La ejecución completa del sistema puede requerir tiempo considerable debido a la naturaleza computacionalmente intensiva de los algoritmos de optimización
- Se recomienda ejecutar los scripts en el orden indicado para garantizar la disponibilidad de datos de entrada
- El sistema incluye logging detallado para monitorización del progreso y detección de errores