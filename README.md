# Challenge MLOps - Predicción de Esperanza de Vida (Cáncer)

Este proyecto implementa un ciclo de vida completo de MLOps, desde la ingesta de datos hasta el despliegue de una API, para predecir la esperanza de vida en pacientes con cáncer.

## Tecnologías Utilizadas

* **Lenguaje:** Python 3.10 (Entorno gestionado con Conda)
* **Control de Versiones de Datos:** DVC (Data Version Control)
* **Seguimiento de Experimentos:** MLflow
* **Modelado:** Scikit-learn (Random Forest Regressor)
* **Despliegue:** FastAPI + Uvicorn
* **Entorno:** Ubuntu / WSL

## Dataset
Se utiliza un dataset de Kaggle con **160,000 registros** y 23 columnas que incluyen:
* Información demográfica (País, Grupo de edad).
* Datos médicos (Tipo de cáncer, Incidencia, Mortalidad).
* Factores de riesgo (Consumo de alcohol, tabaco, obesidad).

## Estructura del Proyecto

* `data/`: Datos crudos y procesados gestionados por DVC.
* `src/`: Scripts de Python para el pipeline:
  * `load_data.py`: Descarga e ingesta.
  * `preprocess.py`: Limpieza de datos y manejo de outliers (IQR).
  * `train.py`: Entrenamiento y registro del modelo en MLflow.
* `notebooks/`: Análisis Exploratorio de Datos (EDA).
* `app/`: Aplicación FastAPI para servir el modelo.

## Pipeline y Reproducibilidad

El proyecto utiliza **DVC** para orquestar las etapas. Para reproducir el flujo completo:

```bash
dvc repro
