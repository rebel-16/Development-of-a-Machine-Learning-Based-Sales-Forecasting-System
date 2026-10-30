import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


DATA_FILE = Path(__file__).resolve().parent / "sales_data.csv"
RANDOM_SEED = 42
START_DATE = date(2024, 1, 1)
NUMBER_OF_DAYS = 730


def generate_sales_data():
    random_generator = random.Random(RANDOM_SEED)

    with DATA_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Date", "Sales"])

        for day_number in range(NUMBER_OF_DAYS):
            sales_date = START_DATE + timedelta(days=day_number)
            yearly_seasonality = 9000 * math.sin(2 * math.pi * day_number / 365.25)
            weekly_seasonality = 2500 * math.sin(2 * math.pi * sales_date.weekday() / 7)
            random_variation = random_generator.gauss(0, 5500)
            sales_amount = max(
                1000,
                50000 + (day_number * 3) + yearly_seasonality + weekly_seasonality + random_variation,
            )
            writer.writerow([sales_date.isoformat(), round(sales_amount, 2)])


if __name__ == "__main__":
    generate_sales_data()
    print(f"Created {NUMBER_OF_DAYS} synthetic daily sales records in {DATA_FILE.name}.")