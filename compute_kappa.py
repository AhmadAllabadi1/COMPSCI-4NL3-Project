import pandas as pd
from sklearn.metrics import cohen_kappa_score

CSV_PATH = "merged_data.csv"

df = pd.read_csv(CSV_PATH)

overlap = df.dropna(subset=["label1", "label2"]).copy()

# Clean labels (strip whitespace, uppercase)
overlap["label1"] = overlap["label1"].astype(str).str.strip().str.upper()
overlap["label2"] = overlap["label2"].astype(str).str.strip().str.upper()

n_overlap = len(overlap)
if n_overlap == 0:
    raise ValueError("No overlapping rows found (no rows with BOTH label1 and label2).")

percent_agreement = (overlap["label1"] == overlap["label2"]).mean()

kappa = cohen_kappa_score(overlap["label1"], overlap["label2"])

print("=== Inter-Annotator Agreement (Cohen’s Kappa) ===")
print(f"Total rows in file: {len(df)}")
print(f"Overlapping (double-annotated) rows used: {n_overlap}")
print(f"Percent agreement: {percent_agreement:.4f} ({percent_agreement*100:.2f}%)")
print(f"Cohen’s Kappa (κ): {kappa:.4f}")

if kappa < 0:
     interpretation = "Worse than chance"
elif kappa < 0.21:
     interpretation = "Slight agreement"
elif kappa < 0.41:
     interpretation = "Fair agreement"
elif kappa < 0.61:
     interpretation = "Moderate agreement"
elif kappa < 0.81:
     interpretation = "Substantial agreement"
else: 
     interpretation = "Almost perfect agreement"

print(f"Interpretation: {interpretation}")