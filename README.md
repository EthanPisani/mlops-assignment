# mlops-assignment
1. Downloaded German Cars dataset from https://www.kaggle.com/datasets/yaminh/german-car-insights to `datasets/`.

2. Explore the data and break it down by year of sale (datasets per year) in `assignment.ipynb`.

3. Created an ML model for price prediction. Using Random Forest Regressor Scikit Learn model. Used ML Flow for experiments and logging in `assignment.ipynb`.

    As part of the model training did some feature extractions
    
    Demonstrate how these features will be extracted and stored if a new batch of data is provided (e.g. use one of the latest year’s data)
    
    Extract 2 sets of features for the model and demonstrate how these will be used in ML Flow, basic and extended features.

4. Wrote a python server to serve the model using FastAPI in `serving/main.py`

5. Built a Dockerfile for containerized deployment of the price prediction API in `Dockerfile`

6. Built a Github Actions Workflow to automatically deploy the model `.github/workflows/build.yml` and deployed with railway.app

7. Implement model monitoring using the Evidently logging framework and accessable at https://mlops-assignment-production.up.railway.app/monitor from file `serving/main.py`

8. The solution can be extended to support batch inference vs real time inference in `serving/main.py` which access realtime single prediction and batch list prediction.



## API Documentation

Base URL: https://mlops-assignment-production.up.railway.app
## Endpoints

### GET `/health`

**Description:** Health check endpoint to verify the API is running.

**Response:**

* `200 OK` — Returns a simple confirmation string.

**Example:**

```bash
curl https://mlops-assignment-production.up.railway.app/health
```

---

### POST `/predict`

**Description:** Predict the price of a single car.

**Request Body:**

```json
{
  "month": 5,
  "year": 2020,
  "power_kw": 100,
  "power_ps": 104,
  "fuel_consumption_l_100km": 6.5,
  "fuel_consumption_g_km": 150,
  "mileage_in_km": 350430,
  "brand": "bmw",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "color": "red"
}
```

**Response:**

* `200 OK` — Returns the predicted price as a string.
* `422 Unprocessable Entity` — Validation error.

**Example:**

```bash
curl -X 'POST' \
  'https://mlops-assignment-production.up.railway.app/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "month": 5,
  "year": 2020,
  "power_kw": 100,
  "power_ps": 104,
  "fuel_consumption_l_100km": 6.5,
  "fuel_consumption_g_km": 150,
  "mileage_in_km": 350430,
  "brand": "bmw",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "color": "red"
}
'
```

---

### POST `/predict_batch`

**Description:** Predict the prices for multiple cars in a batch.

**Request Body:**

```json
{
"cars": [{
    "month": 5,
    "year": 2020,
    "power_kw": 100,
    "power_ps": 104,
    "fuel_consumption_l_100km": 6.5,
    "fuel_consumption_g_km": 150,
    "mileage_in_km": 350430,
    "brand": "bmw",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "color": "red"
    },
    {
    "month": 9,
    "year": 2004,
    "power_kw": 80,
    "power_ps": 184,
    "fuel_consumption_l_100km": 9.5,
    "fuel_consumption_g_km": 140,
    "mileage_in_km": 3430,
    "brand": "bmw",
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "color": "black"
    }]
}
```

**Response:**

* `200 OK` — Returns a list of predicted prices.
* `422 Unprocessable Entity` — Validation error.

**Example:**

```bash
curl -X 'POST' \
  'https://mlops-assignment-production.up.railway.app/predict_batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
        "cars": [{
        "month": 5,
        "year": 2020,
        "power_kw": 100,
        "power_ps": 104,
        "fuel_consumption_l_100km": 6.5,
        "fuel_consumption_g_km": 150,
        "mileage_in_km": 350430,
        "brand": "bmw",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "color": "red"
        },
        {
        "month": 9,
        "year": 2004,
        "power_kw": 80,
        "power_ps": 184,
        "fuel_consumption_l_100km": 9.5,
        "fuel_consumption_g_km": 140,
        "mileage_in_km": 3430,
        "brand": "bmw",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "color": "black"
        }]
      }'
```

---

### GET `/monitor`

**Description:** Generate and return a monitoring report using Evidently.

**Response:**

* `200 OK` — Returns the report as a string (HTML).

**Example:**

```bash
curl https://mlops-assignment-production.up.railway.app/monitor
```

---

## Schema Notes

* All string fields (e.g., `brand`, `color`, `fuel_type`) are case-sensitive and must be from known categories.
* Missing or invalid values will trigger a `422` error.

---

