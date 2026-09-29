import pandas as pd
import numpy as np
import re

# =====================================================================
# TASK 1 — Working with strings
# =====================================================================

# a) Read the cars_autoscout24.csv from this week (extended version) to a pandas data frame
try:
    df = pd.read_csv('Week_LC/LC_03/cars_autoscout24.csv', sep=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('Week_LC/LC_03/cars_autoscout24.csv', sep=';', encoding='latin1')

print("=" * 70)
print("TASK 1 — Working with strings")
print("=" * 70)
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# b) Create a new variable `Str_len` containing the length of each string in the variable `Description`
df['Str_len'] = df['Description'].str.len()
print(f"\n1b) String lengths created:")
print(df[['Description', 'Str_len']].head())

# c) Create a new variable `Description_upper` from the variable `Description` containing only uppercase letters
df['Description_upper'] = df['Description'].str.upper()
print(f"\n1c) Uppercase description created:")
print(df[['Description', 'Description_upper']].head())

# d) Remove all leading and trailing empty spaces in `Description_upper`
df['Description_upper'] = df['Description_upper'].str.strip()
print(f"\n1d) Spaces stripped from Description_upper:")
print(df[['Description_upper']].head())


# =====================================================================
# TASK 2 — Regular expressions (regex)
# =====================================================================

print("\n" + "=" * 70)
print("TASK 2 — Regular expressions (regex)")
print("=" * 70)

# a) Extract the price as numerical value (see last week) and store it in a new variable `Price_numeric`
def extract_numerical_value(price):
    """Extract numerical price value from Price column"""
    if pd.isna(price):
        return None
    digits = str(price).replace("'", '')
    match = re.search(r"\d+", digits)
    return float(match.group()) if match else None

df['Price_numeric'] = df['Price'].apply(extract_numerical_value)
print(f"\n2a) Price_numeric extracted:")
print(df[['Price', 'Price_numeric']].head(10))

# b) Extract the original price (germ.: Neupreis) from `Description_upper` and store it in a variable `Price_original`
def extract_original_price(description):
    """Extract original price (Neupreis) from description using regex"""
    if pd.isna(description):
        return None
    # Look for pattern like "NEUPREIS" followed by numbers (with optional apostrophes)
    match = re.search(r"NEUPREIS\s*[\w\s]*?(\d+(?:'\d+)*)", description)
    if match:
        price_str = match.group(1).replace("'", '')
        try:
            return float(price_str)
        except ValueError:
            return None
    return None

df['Price_original'] = df['Description_upper'].apply(extract_original_price)
print(f"\n2b) Price_original extracted:")
print(df[['Description_upper', 'Price_original']].head(10))
print(f"\nRows with Price_original found: {df['Price_original'].notna().sum()}")

# c) Create a new binary variable `Occasion` with a value of `1` if car type (germ.: Fahrzeugart) is Occasion and a value of `0` otherwise
df['Occasion'] = (df['Car_type'] == 'Occasion').astype(int)
print(f"\n2c) Occasion variable created:")
print(df[['Car_type', 'Occasion']].head(10))
print(f"\nOccasion value counts:\n{df['Occasion'].value_counts()}")


# =====================================================================
# TASK 3 — Working with pivot tables
# =====================================================================

print("\n" + "=" * 70)
print("TASK 3 — Working with pivot tables")
print("=" * 70)

# a) Create a subset of the data frame with all missing values removed
df_clean = df[['Occasion', 'Price_numeric', 'Price_original']].dropna()
print(f"\n3a) Clean subset created (rows with no missing values):")
print(f"Original shape: {df.shape}")
print(f"Clean subset shape: {df_clean.shape}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")

# b) Create a pivot table with:
#    - `Occasion` as index variable,
#    - `Price_numeric` and `Price_original` as values
#    - `np.mean` (i.e. mean from the numpy library) as the aggregation function
pivot_table = df_clean.pivot_table(
    index='Occasion',
    values=['Price_numeric', 'Price_original'],
    aggfunc=np.mean
)

print(f"\n3b) Pivot table created:")
print(pivot_table)

# c) Report the actual mean `Price_numeric` and `Price_original` values your
#    pivot table shows for Occasion = 0 and Occasion = 1
print("\n3c) Mean values from pivot table:")
print("\n" + "-" * 70)
print("FINAL RESULTS:")
print("-" * 70)

for occasion in [0, 1]:
    occasion_label = "Occasion" if occasion == 1 else "New car (not Occasion)"
    price_numeric_mean = pivot_table.loc[occasion, 'Price_numeric']
    price_original_mean = pivot_table.loc[occasion, 'Price_original']
    
    print(f"\nOccasion = {occasion} ({occasion_label}):")
    print(f"  Mean Price_numeric: CHF {price_numeric_mean:,.2f}")
    print(f"  Mean Price_original: CHF {price_original_mean:,.2f}")

print("\n" + "=" * 70)
