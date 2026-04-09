from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Cargar modelo
mlflow.set_tracking_uri("file:./mlruns") # para que fastAPI pueda encontrar el run
RUN_ID = '8d39d1b914f1485f996057bf2d7ebd2c'
MODEL_URI = f"runs:/{RUN_ID}/random_forest_model"
model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def home():
    return {"message": "API funcionando"}


@app.post("/predict")
def predict(data: dict):
    """
    Espera un JSON con features
    """
    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {
        "prediction": prediction.tolist()
    }




# uvicorn app.main:app --reload
# abre http://127.0.0.1:8000/docs (swager automático)

# Para probar con curl:
"""
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "Country":"Turkey",
  "Age_Group":"15-24",
  "Cancer_Type":"Lung",
  "Risk_Factors":"Obesity",
  "Incidence":44,
  "Mortality":457,
  "Prevalence":955,
  "Urban_Population":32.90675830042905,
  "Health_Expenditure_%GDP":11.83400529544704,
  "Tobacco_Use_%":12.578421085596268,
  "Alcohol_Consumption_Liters":1.4459093633093378,
  "Physical_Activity_%":33.19719664120764,
  "Obesity_%":33.944195813190674,
  "Air_Quality_Index":96,
  "UV_Radiation":2.877395416241618,
  "Family_History_%":44.30086194602105,
  "Genetic_Mutation_%":6.924822037196543,
  "Treatment_Coverage_%":97.21091218546712,
  "GDP_per_Capita":29779,
  "Health_Infrastructure_Index":2.3241390992745297,
  "Education_Index":0.8754517690807012,
}'"Population_Density":736.6090059054068
"""

# Devuelve:
# {"prediction":[67.36826897984962]}
