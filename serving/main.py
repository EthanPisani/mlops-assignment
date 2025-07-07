from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import mlflow.pyfunc
import logging
from datetime import datetime
import whylogs as why

# === Load MLflow model ===
MODEL_URI = "mlruns/176985576620168457/models/m-b69e7559fbd04823b52d46c3ad0fd693/artifacts"
model = mlflow.pyfunc.load_model(MODEL_URI)

# === Initialize WhyLogs logger ===
why_logger = why.logger(log_args={"dataset_id": "car_price_inference"})

# === Define categories ===
KNOWN_BRANDS = [
    "aston-martin", "audi", "bentley", "bmw", "cadillac", "chevrolet", "chrysler", "citroen", "dacia", "daewoo",
    "daihatsu", "dodge", "ferrari", "fiat", "ford", "honda", "hyundai", "infiniti", "isuzu", "jaguar", "jeep", "kia",
    "lada", "lamborghini", "lancia", "land-rover", "maserati", "mazda"
]
KNOWN_FUEL_TYPES = [
    "Diesel", "Diesel Hybrid", "Electric", "Ethanol", "Hybrid", "Hydrogen", "LPG", "Other", "Petrol"
]
KNOWN_TRANSMISSIONS = ["Manual", "Semi-automatic", "Unknown"]
KNOWN_COLORS = [
    "black", "blue", "bronze", "brown", "gold", "green", "grey", "orange", "red", "silver", "violet", "white", "yellow"
]

NUMERIC_COLUMNS = ["month", "year", "power_kw", "power_ps", "fuel_consumption_l_100km", "fuel_consumption_g_km", "mileage_in_km"]

FEATURE_COLUMNS = [
    *NUMERIC_COLUMNS,
    *[f"brand_{b}" for b in KNOWN_BRANDS],
    *[f"fuel_type_{f}" for f in KNOWN_FUEL_TYPES],
    *[f"transmission_type_{t}" for t in KNOWN_TRANSMISSIONS],
    *[f"color_{c}" for c in KNOWN_COLORS]
]

# === Define Input Model ===
class CarInput(BaseModel):
    month: int
    year: int
    power_kw: float
    power_ps: float
    fuel_consumption_l_100km: float
    fuel_consumption_g_km: float
    mileage_in_km: float
    brand: str
    fuel_type: str
    transmission_type: str
    color: str

app = FastAPI(title="Car Price Predictor API with WhyLogs")

def validate_category(value: str, valid_list: List[str], field_name: str):
    if value not in valid_list:
        raise ValueError(f"Invalid {field_name}: {value}. Valid options are: {valid_list}")
    return value

# health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}
@app.post("/predict")
async def predict_price(car_input: CarInput):
    try:
        # Validate input categories
        brand = validate_category(car_input.brand, KNOWN_BRANDS, "brand")
        fuel = validate_category(car_input.fuel_type, KNOWN_FUEL_TYPES, "fuel_type")
        transmission = validate_category(car_input.transmission_type, KNOWN_TRANSMISSIONS, "transmission_type")
        color = validate_category(car_input.color, KNOWN_COLORS, "color")

        # Create base input row
        input_data = {col: 0 for col in FEATURE_COLUMNS}
        for col in NUMERIC_COLUMNS:
            input_data[col] = getattr(car_input, col)

        # Set one-hot encodings
        input_data[f"brand_{brand}"] = 1
        input_data[f"fuel_type_{fuel}"] = 1
        input_data[f"transmission_type_{transmission}"] = 1
        input_data[f"color_{color}"] = 1

        input_df = pd.DataFrame([input_data])
        print(f"Processed input: {input_df}")

        prediction = model.predict(input_df)[0]

        # Log with WhyLogs
        input_df["prediction"] = prediction
        # why_logger.log_dataframe(input_df, name="car_price_prediction", timestamp=datetime.now())
        # log to file for now
        with open("whylogs_output.txt", "a") as f:
            f.write(f"{datetime.now()}: {input_df.to_dict(orient='records')}\n")
        # limit to 2 decimal places
        prediction = round(prediction, 2)
        logging.info(f"Prediction made: {prediction}")
        

        return {"predicted_price": prediction}

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "Internal server error"}
