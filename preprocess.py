import pandas as pd
from datetime import datetime

# Load patient data
df = pd.read_csv("patients.csv")

# Convert birthdate to datetime
df["BIRTHDATE"] = pd.to_datetime(
    df["BIRTHDATE"],
    errors="coerce"
)

# Calculate age
today = pd.Timestamp.today()

df["AGE"] = (
    today.year
    - df["BIRTHDATE"].dt.year
    - (
        (
            df["BIRTHDATE"].dt.month > today.month
        )
        |
        (
            (df["BIRTHDATE"].dt.month == today.month)
            & (df["BIRTHDATE"].dt.day > today.day)
        )
    ).astype(int)
)

# Calculate expense-to-coverage ratio
df["EXPENSE_COVERAGE_RATIO"] = (
    df["HEALTHCARE_EXPENSES"]
    / df["HEALTHCARE_COVERAGE"].replace(0, 1)
)

# Select useful recommendation features
member_df = df[
    [
        "Id",
        "AGE",
        "MARITAL",
        "GENDER",
        "CITY",
        "STATE",
        "COUNTY",
        "ZIP",
        "LAT",
        "LON",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE",
        "EXPENSE_COVERAGE_RATIO"
    ]
].copy()

# Save processed dataset
member_df.to_csv(
    "processed_members.csv",
    index=False
)

print("\nProcessed Member Data:")
print("======================")
print(member_df)

print("\nTotal members:", len(member_df))
print("\nProcessed dataset saved as: processed_members.csv")