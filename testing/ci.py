# runs tests against the service

import requests

#  curl -X 'POST' \
# 'http://localhost:4723/predict' \
# -H 'accept: application/json' \
# -H 'Content-Type: application/json' \
# -d '{ \
# "month": 5, \
# "year": 2020, \
# "power_kw": 100, \
# "power_ps": 104, \
# "fuel_consumption_l_100km": 6.5, \
# "fuel_consumption_g_km": 150, \
# "mileage_in_km": 350430, \
# "brand": "bmw", \
# "fuel_type": "Petrol", \
# "transmission_type": "Manual", \
# "color": "red" \
# }'

def test_predict():
    url = "http://localhost:4723/predict"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
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
    # {"predicted_price":30290.04} <--- expected response
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    assert "predicted_price" in response.json(), "Response JSON does not contain 'predicted_price' key"
    assert isinstance(response.json()["predicted_price"], (int, float)), "Predicted price is not a number"
    assert response.json()["predicted_price"] > 0, "Predicted price should be a positive number"
    assert response.json()["predicted_price"] < 100000, "Predicted price seems unreasonably high"
    print(response.json())  # For debugging purposes

if __name__ == "__main__":
    test_predict()
    print("Test passed successfully!")