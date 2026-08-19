import pandas as pd

# Load processed patient data
members = pd.read_csv("processed_members.csv")

# Load insurance plan data
plans = pd.read_csv("plans.csv")

print("\n===== MEMBERS =====")
print(members.head())

print("\n===== PLANS =====")
print(plans)

print("\nNumber of members:", len(members))
print("Number of plans:", len(plans))