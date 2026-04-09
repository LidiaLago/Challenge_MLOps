import os
import pandas as pd
import kagglehub

def download_data():
    # Descargar dataset desde Kaggle (devuelve carpeta temporal)
    path = kagglehub.dataset_download("ankushpanday1/cancer-datasettop-50-populated-countries")
    print("Dataset descargado en carpeta temporal:", path)

    # Buscar el primer CSV dentro de la carpeta descargada
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset.")

    csv_path = os.path.join(path, csv_files[0])

    # Leer el CSV con pandas
    df = pd.read_csv(csv_path)

    # Crear carpeta destino fuera de src
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/raw")
    os.makedirs(output_dir, exist_ok=True)

    # Guardar CSV en data/raw/
    output_path = os.path.join(output_dir, "cancer_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"Datos guardados en {output_path}")

if __name__ == "__main__":
    download_data()