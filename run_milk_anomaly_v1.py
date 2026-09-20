import pandas as pd
import numpy as np
import os

def run_milk_anomaly_system(input_csv, output_dir="output"):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load robot CSV
    df = pd.read_csv(input_csv, sep=";", encoding="latin1")

    # 2. Fix milk yield values
    df["Latte. kg"] = (
        df["Latte. kg"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("\t", "", regex=False)
        .astype(float)
    )

    # 3. Rename columns
    df = df.rename(columns={
        "Vacca": "cow_id",
        "Data": "date",
        "Latte. kg": "milk_yield"
    })

    # 4. Convert date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # 5. Keep only successful milkings
    df = df[df["Tipo Mung"] == "Ok"]
    df = df[df["milk_yield"] > 0]

    # 6. Daily aggregation (median per milking)
    daily = (
        df
        .groupby(["cow_id", "date"])["milk_yield"]
        .median()
        .reset_index()
    )

    # 7. Cow-specific Z-score
    daily["milk_z"] = (
        daily
        .groupby("cow_id")["milk_yield"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    # 8. Anomaly flags
    daily["milk_anomaly"] = daily["milk_z"].abs() > 2
    daily["low_milk"] = daily["milk_z"] < -2
    daily["high_milk"] = daily["milk_z"] > 2

    # 9. Cow-level summary
    cow_summary = (
        daily
        .groupby("cow_id")
        .agg(
            days_observed=("milk_yield", "count"),
            anomaly_days=("milk_anomaly", "sum"),
            low_milk_days=("low_milk", "sum"),
            avg_milk=("milk_yield", "mean"),
            std_milk=("milk_yield", "std")
        )
        .reset_index()
    )

    # 10. Filter meaningful problem cows
    problem_cows = cow_summary[
        (cow_summary["days_observed"] >= 10) &
        (cow_summary["low_milk_days"] >= 2)
    ]

    # 11. Save outputs
    daily.to_csv(
        os.path.join(output_dir, "daily_milk_anomalies.csv"),
        index=False
    )

    problem_cows.to_csv(
        os.path.join(output_dir, "problem_cows_summary.csv"),
        index=False
    )

    print("Milk anomaly analysis completed.")
    print("Results saved in:", output_dir)

if __name__ == "__main__":
    import os

    input_dir = "input"
    output_dir = "output"

    # Find all CSV files in input directory
    csv_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    ]

    if len(csv_files) == 0:
        print("❌ No CSV file found in 'input/' folder.")
        print("Please place ONE robot CSV file in the input folder.")
    
    elif len(csv_files) > 1:
        print("❌ Multiple CSV files found in 'input/' folder:")
        for f in csv_files:
            print(" -", f)
        print("Please keep only ONE CSV file in the input folder.")
    
    else:
        input_csv = os.path.join(input_dir, csv_files[0])
        print(f"📄 Using input file: {csv_files[0]}")

        run_milk_anomaly_system(
            input_csv=input_csv,
            output_dir=output_dir
        )
