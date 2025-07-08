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

headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

# === NEW === Batch test function
def test_batch_predict(batch_size=10):
    url = "http://localhost:4723/predict_batch"
    inputs = [generate_random_input() for _ in range(batch_size)]
    data = {"cars": inputs}
    start_timer = time.perf_counter()
    response = requests.post(url, headers=headers, json=data)
    end_timer = time.perf_counter()
    print(f"Response time Batch: {end_timer - start_timer:.4f} seconds")
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    result = response.json()
    assert "predicted_prices" in result, "Response missing 'predicted_prices'"
    predictions = result["predicted_prices"]

    assert isinstance(predictions, list), "predicted_prices should be a list"
    assert len(predictions) == batch_size, f"Expected {batch_size} predictions, got {len(predictions)}"

    for i, price in enumerate(predictions):
        assert isinstance(price, (int, float)), f"Prediction {i} is not a number"
        assert 0 < price < 100000, f"Prediction {i} out of reasonable range: {price}"
        print(f"[Batch {i+1}] ✅ {price:.2f} for {inputs[i]['brand']} ({inputs[i]['year']})")

    print(f"\n✔️ Batch test passed for {batch_size} inputs")

# Reuse random input generator
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

# Fixed single request test
def test_predict():
    url = "http://localhost:4723/predict"
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
    start_timer = time.perf_counter()
    response = requests.post(url, headers=headers, json=data)
    end_timer = time.perf_counter()
    print(f"Response time Single: {end_timer - start_timer:.4f} seconds")
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    assert "predicted_price" in response.json(), "Response JSON does not contain 'predicted_price' key"
    assert isinstance(response.json()["predicted_price"], (int, float)), "Predicted price is not a number"
    assert 0 < response.json()["predicted_price"] < 100000, "Predicted price out of expected range"
    print(f"✅ Fixed test passed: {response.json()}")

# Random individual request loop
def test_random_requests(n=100):
    url = "http://localhost:4723/predict"
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
            print(f"[{i+1}] ❌ Failed response: {response.status_code} {response.text}")
        time.sleep(0.1)
    print(f"\n✔️ {success_count}/{n} random requests succeeded")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run tests for ML model prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("predict", help="Run a single test prediction")

    subparsers.add_parser("random", help="Run 100 randomized single POST requests")

    batch_parser = subparsers.add_parser("batch", help="Run batch test")
    batch_parser.add_argument("--size", type=int, required=True, help="Batch size")

    args = parser.parse_args()

    if args.command == "predict":
        test_predict()
    elif args.command == "random":
        print("\nStarting 100 randomized single POST requests...")
        test_random_requests(100)
    elif args.command == "batch":
        print(f"\nStarting a batch test with {args.size} inputs...")
        test_batch_predict(batch_size=args.size)

    print("\nTest(s) completed successfully!")

if __name__ == "__main__":
    main()

