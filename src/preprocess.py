import os
import pandas as pd


def remove_outliers(df, columns=None):
    df_clean = df.copy()

    if columns is None:
        columns = df_clean.select_dtypes(include=["number"]).columns

    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[
            (df_clean[column] >= lower_bound) &
            (df_clean[column] <= upper_bound)
        ]

    return df_clean


def preprocess_data():
    # Ruta
    base_dir = os.path.dirname(os.path.dirname(__file__))

    raw_path = os.path.join(base_dir, "data/raw/cancer_dataset.csv")
    processed_path = os.path.join(base_dir, "data/processed/clean_cancer_dataset.csv")

    # Cargar datos
    df = pd.read_csv(raw_path)
    print(f"Shape inicial: {df.shape}")

    # Limpieza
    df = df.drop_duplicates()
    df = df.dropna()

    # Outliers (solo numéricas)
    df = remove_outliers(df)

    print(f"Shape tras limpieza: {df.shape}")

    # Guardar
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Dataset limpio guardado en: {processed_path}")


if __name__ == "__main__":
    preprocess_data()