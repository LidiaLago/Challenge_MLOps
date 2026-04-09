import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Life_Expectancy_Optimization")


def run_experiment(params):

    df = pd.read_csv("data/processed/clean_cancer_dataset.csv")

    X = df.drop(columns=["Life_Expectancy"])
    y = df["Life_Expectancy"]

    # Detectar tipos automáticamente
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocesamiento
    preprocessor = ColumnTransformer([
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():

        # Log parámetros
        mlflow.log_param("model", "RandomForest")
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Predicción
        preds = pipeline.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log modelo
        mlflow.sklearn.log_model(
            pipeline,
            name="random_forest_model"
        )

        print(f"Params: {params}")
        print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")
        print("-" * 50)


# GRID
if __name__ == "__main__":

    param_grid = [
        {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        },
        {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        },
        {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "log2"
        },
        {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        }
    ]

    for params in param_grid:
        print(f"Iniciando experimento con: {params}")
        run_experiment(params)