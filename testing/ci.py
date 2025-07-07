import requests
import random
import time

# Valid categorical options
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

url = "http://localhost:4723/predict"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# Fixed test case
def test_predict():
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
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    assert "predicted_price" in response.json(), "Response JSON does not contain 'predicted_price' key"
    assert isinstance(response.json()["predicted_price"], (int, float)), "Predicted price is not a number"
    assert response.json()["predicted_price"] > 0, "Predicted price should be a positive number"
    assert response.json()["predicted_price"] < 100000, "Predicted price seems unreasonably high"
    print(f"Fixed test passed: {response.json()}")

# Random test generator
def generate_random_input():
    return {
        "month": random.randint(1, 12),
        "year": random.randint(1995, 2024),
        "power_kw": round(random.uniform(30, 300), 1),
        "power_ps": round(random.uniform(40, 350), 1),
        "fuel_consumption_l_100km": round(random.uniform(2, 15), 2),
        "fuel_consumption_g_km": round(random.uniform(50, 300), 1),
        "mileage_in_km": random.randint(10000, 400000),
        "brand": random.choice(KNOWN_BRANDS),
        "fuel_type": random.choice(KNOWN_FUEL_TYPES),
        "transmission_type": random.choice(KNOWN_TRANSMISSIONS),
        "color": random.choice(KNOWN_COLORS),
    }

def test_random_requests(n=100):
    success_count = 0
    for i in range(n):
        data = generate_random_input()
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200 and "predicted_price" in response.json():
            price = response.json()["predicted_price"]
            if isinstance(price, (int, float)) and 0 < price < 100000:
                print(f"[{i+1}] ✅ {price} - {data['brand']} ({data['year']})")
                success_count += 1
            else:
                print(f"[{i+1}] ❌ Invalid price: {price}")
        else:
            print(f"[{i+1}] ❌ Failed response: {response.status_code} {response.text}")
        time.sleep(0.1)  # slight delay to prevent overload
    print(f"\n✔️ {success_count}/{n} requests succeeded")

if __name__ == "__main__":
    test_predict()
    print("Starting 100 randomized test requests...")
    test_random_requests(100)
