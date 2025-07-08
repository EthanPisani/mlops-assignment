# 4. Serving model with FastAPI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow.pyfunc
import logging
from datetime import datetime, timezone
import json
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 

# === Load MLflow model ===
MODEL_URI = "mlruns/176985576620168457/models/m-b69e7559fbd04823b52d46c3ad0fd693/artifacts"
model = mlflow.pyfunc.load_model(MODEL_URI)

# === Define constants ===
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

app = FastAPI(title="Car Price Predictor API with Evidently")

def validate_category(value: str, valid_list: List[str], field_name: str):
    if value not in valid_list:
        raise ValueError(f"Invalid {field_name}: {value}. Valid options are: {valid_list}")
    return value

@app.get("/health")
async def health():
    return {"status": "ok"}

# real-time prediction endpoint
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
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        # Evidently logging
        input_df["prediction"] = prediction
        input_df["timestamp"] = datetime.now(timezone.utc).timestamp()

        # Log raw input + prediction to JSONL
        with open("evidently_input_log.jsonl", "a") as log_file:
            log_file.write(json.dumps(input_df.to_dict(orient="records")[0]) + "\n")

        logging.info(f"Prediction made: {prediction}")
        return {"predicted_price": prediction}

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "Internal server error"}


class CarInputBatch(BaseModel):
    cars: List[CarInput]

# 8. Batch prediction endpoint
@app.post("/predict_batch")
async def predict_batch(car_inputs: CarInputBatch):
    try:
        rows = []

        for car_input in car_inputs.cars:
            # Validate input
            brand = validate_category(car_input.brand, KNOWN_BRANDS, "brand")
            fuel = validate_category(car_input.fuel_type, KNOWN_FUEL_TYPES, "fuel_type")
            transmission = validate_category(car_input.transmission_type, KNOWN_TRANSMISSIONS, "transmission_type")
            color = validate_category(car_input.color, KNOWN_COLORS, "color")

            # Prepare row
            input_data = {col: 0 for col in FEATURE_COLUMNS}
            for col in NUMERIC_COLUMNS:
                input_data[col] = getattr(car_input, col)

            input_data[f"brand_{brand}"] = 1
            input_data[f"fuel_type_{fuel}"] = 1
            input_data[f"transmission_type_{transmission}"] = 1
            input_data[f"color_{color}"] = 1
            rows.append(input_data)

        input_df = pd.DataFrame(rows)
        predictions = model.predict(input_df)
        predictions = [round(float(p), 2) for p in predictions]

        # Logging each record 
        input_df["prediction"] = predictions
        input_df["timestamp"] = datetime.now(timezone.utc).timestamp()

        with open("evidently_input_log.jsonl", "a") as log_file:
            for record in input_df.to_dict(orient="records"):
                log_file.write(json.dumps(record) + "\n")

        return {"predicted_prices": predictions}

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "Internal server error"}

# 7. Evidently for monitoring
@app.get("/monitor", response_class=HTMLResponse)
async def generate_monitoring_report():
    try:
        df = pd.read_json("evidently_input_log.jsonl", lines=True)
        report = Report(metrics=[DataDriftPreset()])
        reference = df.head(50)
        current = df.tail(50)
        snapshot = report.run(reference_data=reference, current_data=current)
        html_report = snapshot.get_html_str(as_iframe=False)
        with open("monitor_report.html", "w") as report_file:
            report_file.write(html_report)
        # return the html
        return HTMLResponse(content=html_report, status_code=200)

    except Exception as e:
        logging.error(f"Monitoring error: {e}")
        return {"error": str(e)}
