# Cleaning_and_Validating_Healthcare_Data_Using_Python.ipynb
This Repository is about Cleaning and Validating HealthCare Data Using python that I did after attending a Coursera class on Foundations of HealthCare Data Analytics class on 
# Cleaning and Validating Healthcare Data Using Python
Time estimate: **30** minutes
## Objectives

After completing this lab, you will be able to:

 - Identify and handle common data quality issues including missing values, duplicates, inconsistent entries, and outliers.
 - Remove Personally Identifiable Information (PII) to ensure data privacy compliance with HIPAA and GDPR.
 - Apply data transformation techniques including standardization, normalization, and encoding.


## What you will do in this lab

In this hands-on lab, you will work with real-world healthcare data that contains typical quality issues found in medical datasets. You'll learn to clean and prepare this data for machine learning applications while maintaining patient privacy.

You will:

- Explore and identify data quality issues in a healthcare dataset
- Remove sensitive patient information (PII) to comply with privacy regulations
- Handle missing values using appropriate imputation strategies
- Standardize inconsistent categorical data and mixed units
- Detect and handle outliers using statistical methods
- Engineer meaningful features such as BMI and temporal indicators
- Encode categorical variables and scale numeric features for ML readiness

## Overview

Data preprocessing is a critical step in any machine learning pipeline, but it becomes especially important in healthcare applications where data quality directly impacts patient outcomes. Raw healthcare data often contains inconsistencies, missing values, mixed formats, and privacy-sensitive information that must be carefully addressed before building predictive models.

In this lab, you'll work with a synthetic healthcare dataset that simulates real-world challenges such as inconsistent gender labels (M/Male/F/Female), mixed measurement units (lbs/kg), various date formats, and missing diagnostic information. You'll learn systematic approaches to clean this data while maintaining its utility for analysis.

The preprocessing pipeline you'll build follows industry best practices: first removing PII for privacy, then addressing data quality issues, followed by feature engineering to create more informative variables, and finally transforming the data into a format suitable for machine learning algorithms. These skills are directly applicable to real healthcare analytics projects where clean, privacy-compliant data is essential.

By the end of this lab, you'll have a complete understanding of how to transform messy healthcare data into a clean, standardized dataset ready for predictive modeling tasks such as risk assessment or disease diagnosis.
## About the dataset

This lab uses a synthetic healthcare dataset designed to simulate real-world medical data challenges.

### Dataset overview

The dataset contains patient health records including demographics, vital measurements, diagnostic information, and risk indicators. This data simulates what you might encounter in electronic health records (EHR) systems, complete with the messiness and inconsistencies typical of real medical data. The dataset includes 200 patient records with intentionally introduced quality issues such as missing values, inconsistent formatting, mixed units, and duplicate entries to provide realistic preprocessing practice.

### Column descriptions

1. **Patient_ID** - Unique identifier for each patient (e.g., P016, P_new_126)
2. **Age** - Age of the patient in years (may contain missing values or outliers)
3. **Gender** - Gender of the patient (inconsistent formats: M, Male, F, Female, Other)
4. **Ethnicity** - Ethnic background of the patient (Asian, African, Caucasian, Hispanic with inconsistent capitalization)
5. **Weight** - Weight of the patient in mixed units (kg or lbs, e.g., 70, 150lbs)
6. **Height_cm** - Height of the patient in centimeters (numeric values)
7. **Diagnosis_Date** - Date when diagnosis was made (multiple date formats: YYYY-MM-DD, DD/MM/YYYY)
8. **Diagnosis_Code** - Medical diagnosis code or abbreviation (DEP=Depression, OCD=Obsessive Compulsive Disorder, ANX=Anxiety, ANXITY=typo for Anxiety)
9. **Glucose_mg_dL** - Blood glucose level in mg/dL (may indicate diabetes risk)
10. **Risk** - Binary risk indicator (0 = low risk, 1 = high risk for adverse health outcomes)
11. **Patient_Name** - Full name of the patient (PII - to be removed)
12. **EmailID** - Email address of the patient (PII - to be removed)
## Setup

### Installing required libraries

The following libraries are required to run this lab. Pandas will be used for data manipulation, NumPy for numerical operations, SciPy for statistical functions, and Scikit-learn for preprocessing utilities.
# Install the libraries required for this lab
!pip install pandas
!pip install numpy
!pip install scipy
!pip install scikit-learn
# Optional: suppress warnings for cleaner output
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
### Importing required libraries
import pandas as pd   # For data loading, manipulation, cleaning, and saving (DataFrame operations)
import numpy as np    # For numerical operations, array handling, and missing value operations
import re             # For regular expression pattern matching in text processing

from datetime import datetime, timedelta  # For parsing and manipulating date/time values
import random  # For generating random values during data exploration

from scipy import stats  # For statistical functions like z-score for outlier detection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# StandardScaler: standardizes features (mean=0, std=1) for ML algorithms
# MinMaxScaler: scales features to a fixed range (typically 0-1)
# OneHotEncoder: converts categorical features into binary indicator variables

print("All libraries imported successfully!")
print("Ready to begin healthcare data preprocessing.")
## Step 1: Load and explore the raw data

Before cleaning data, it's essential to understand what you're working with. In this step, you'll load the healthcare dataset and perform an initial exploration to identify data quality issues. This exploration phase helps you make informed decisions about which preprocessing techniques to apply.
# Load the raw healthcare data from CSV file
df = pd.read_csv("https://foundations-of-healthcare-data-analytics-4e579d.gitlab.io/labs/Cleaning_and_Validating_Healthcare_Data_Using_Python/raw_data.csv")

# Display the first few rows to get an initial sense of the data
print("First 5 rows of the dataset:")
df.head()
## Step 2: Understand data quality issues

Real-world healthcare data commonly suffers from four main quality issues:

1. **Missing data**: Important fields left blank or null
2. **Duplicates**: Identical records appearing multiple times
3. **Inconsistent entries**: Same category with different labels (e.g., M vs Male)
4. **Outliers**: Extreme or impossible values (e.g., Age=200, Glucose=500)

Let's systematically identify these issues in the dataset.
# Basic dataset structure
print("Dataset dimensions:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nColumn names and data types:")
print(df.dtypes)
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nPercentage of missing values:")
print((df.isnull().sum() / len(df) * 100).round(2))
# Check for duplicate rows
dup_rows = df.duplicated(keep=False)
print(f"\nNumber of duplicate rows: {dup_rows.sum()}")
if dup_rows.any():
    display(df[dup_rows])
# Identify inconsistent categorical entries
print("\nUnique values in Gender column:")
print(df['Gender'].unique())
print("\nUnique values in Ethnicity column:")
print(df['Ethnicity'].unique())
print("\nUnique values in Diagnosis_Code column:")
print(df['Diagnosis_Code'].unique())
# Check for mixed units in Weight column
print("\nUnique Weight values (showing mixed units):")
display(df['Weight'].unique())
# Statistical summary to identify potential outliers
print("\nStatistical summary of numeric columns:")
display(df[['Age', 'Glucose_mg_dL']].describe())
# Check date format inconsistencies
print("\nSample of Diagnosis_Date values (showing mixed formats):")
display(df['Diagnosis_Date'].sample(10, random_state=42))
# Check class balance for the target variable
print("\nRisk value distribution:")
print(df['Risk'].value_counts())
print("\nRisk percentage distribution:")
print((df['Risk'].value_counts() / len(df) * 100).round(2))
## Step 3: Detect outliers using statistical methods

Outliers can significantly impact machine learning models. You'll use two common statistical methods to detect them:

### Interquartile Range (IQR) method
- **Formula**: IQR = Q3 − Q1 (difference between 75th and 25th percentiles)
- **Outlier definition**: Values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR
- **Best for**: Non-normally distributed data (robust against skewness)

### Z-Score method
- **Formula**: Z = (Value − Mean) / Standard Deviation
- **Outlier definition**: |Z-score| > 3 (more than 3 standard deviations from mean)
- **Best for**: Normally distributed data
# Function to detect outliers using IQR method
def iqr_outliers(series):
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Parameters:
    series: pandas Series - numeric column to check for outliers

    Returns:
    pandas Series - containing only the outlier values
    """
    q1 = series.quantile(0.25)  # 25th percentile
    q3 = series.quantile(0.75)  # 75th percentile
    iqr = q3 - q1                # Interquartile range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return series[(series < lower_bound) | (series > upper_bound)]

# Detect outliers in Age
print("Age outliers (IQR method):")
age_outliers = iqr_outliers(df['Age'].dropna())
print(f"Found {len(age_outliers)} outliers in Age")
print(f"Outlier values: {age_outliers.unique()}")

# Detect outliers in Glucose
print("\nGlucose outliers (IQR method):")
glucose_outliers = iqr_outliers(df['Glucose_mg_dL'].dropna())
print(f"Found {len(glucose_outliers)} outliers in Glucose_mg_dL")
if len(glucose_outliers) > 0:
    print(f"Outlier values: {glucose_outliers.unique()}")
## Step 4: Create a clean copy and remove PII

Privacy protection is paramount in healthcare data. **Personally Identifiable Information (PII)** includes any data that can directly or indirectly identify an individual. Common PII in healthcare includes:

- **Direct identifiers**: Patient names, email addresses, phone numbers, addresses
- **Semi-identifiers**: Patient IDs (can be kept if properly anonymized)
- **Sensitive dates**: Birth dates, exact diagnosis dates (often generalized)

Regulations like **HIPAA** (USA) and **GDPR** (Europe) require removing or anonymizing PII before data analysis or sharing.

You'll create a copy of the original data (to preserve the raw data) and remove PII columns.
# Create a working copy - keep original data untouched for reference
df_clean = df.copy()
print("Working copy created. Original data preserved.")
# Identify and remove PII columns
pii_columns = ['Patient_Name', 'EmailID']
print(f"Removing PII columns: {pii_columns}")

df_clean = df_clean.drop(columns=pii_columns, errors='ignore')

print("\nColumns after removing PII:")
print(df_clean.columns.tolist())
print(f"\nReduced from {len(df.columns)} to {len(df_clean.columns)} columns")
## Step 5: Remove duplicate rows

Duplicate records can occur due to data entry errors, system glitches, or merging datasets. They can:
- Bias analysis by overrepresenting certain patients
- Inflate dataset size artificially
- Cause data leakage in train-test splits

You'll identify and remove exact duplicate rows, keeping only the first occurrence.
# Count duplicates before removal
duplicates_before = df_clean.duplicated().sum()
rows_before = len(df_clean)

# Remove exact duplicate rows (keep first occurrence)
df_clean = df_clean.drop_duplicates(keep='first')

# Report results
rows_after = len(df_clean)
print(f"Rows before deduplication: {rows_before}")
print(f"Duplicate rows found: {duplicates_before}")
print(f"Rows after deduplication: {rows_after}")
print(f"Rows removed: {rows_before - rows_after}")
## Step 6: Standardize inconsistent categorical variables

Inconsistent categorical data is common in healthcare due to:
- Multiple data entry personnel with different conventions
- Data merging from different systems
- Typos and abbreviations

You need to standardize:
- **Gender**: Convert M/Male/m → 'Male', F/Female/f → 'Female'
- **Ethnicity**: Standardize capitalization (asian/Asian/ASIAN → 'Asian')
- **Diagnosis_Code**: Fix typos and standardize (OCD/ocd → 'OCD', ANXITY → 'ANX')

This ensures categorical data is **clean, consistent, and machine-readable**.
# Standardize Gender column
print("Before standardization - Gender unique values:")
print(df_clean['Gender'].value_counts(dropna=False))

# Convert to lowercase and strip whitespace for consistent matching
df_clean['Gender'] = df_clean['Gender'].astype(str).str.strip().str.lower()

# Define mapping for known variations
gender_map = {
    'male': 'Male', 'm': 'Male',
    'female': 'Female', 'f': 'Female',
    'other': 'Other',
    'nan': np.nan, 'none': np.nan
}

# Apply mapping
df_clean['Gender'] = df_clean['Gender'].replace({'nan': np.nan})
df_clean['Gender'] = df_clean['Gender'].map(
    lambda x: gender_map.get(x, x.capitalize() if pd.notna(x) else x)
)

print("\nAfter standardization - Gender unique values:")
print(df_clean['Gender'].value_counts(dropna=False))
# Standardize Ethnicity column
print("Before standardization - Ethnicity unique values:")
print(df_clean['Ethnicity'].value_counts(dropna=False))

# Standardize capitalization
df_clean['Ethnicity'] = df_clean['Ethnicity'].astype(str).str.strip()
df_clean['Ethnicity'] = df_clean['Ethnicity'].replace({'nan': np.nan})
df_clean['Ethnicity'] = df_clean['Ethnicity'].where(
    df_clean['Ethnicity'].isna(),
    df_clean['Ethnicity'].str.capitalize()
)

print("\nAfter standardization - Ethnicity unique values:")
print(df_clean['Ethnicity'].value_counts(dropna=False))
# Standardize Diagnosis_Code column
print("Before standardization - Diagnosis_Code unique values:")
print(df_clean['Diagnosis_Code'].value_counts(dropna=False))

# Convert to uppercase and fix common typos
df_clean['Diagnosis_Code'] = df_clean['Diagnosis_Code'].astype(str).str.strip().str.upper()
df_clean['Diagnosis_Code'] = df_clean['Diagnosis_Code'].replace({
    'NAN': np.nan,
    'ANXITY': 'ANX',  # Fix typo
    'OCD.': 'OCD'      # Remove trailing period
})

print("\nAfter standardization - Diagnosis_Code unique values:")
print(df_clean['Diagnosis_Code'].value_counts(dropna=False))
# Fill missing categorical values with 'Unknown'
df_clean['Gender'] = df_clean['Gender'].fillna('Unknown')
df_clean['Ethnicity'] = df_clean['Ethnicity'].fillna('Unknown')
df_clean['Diagnosis_Code'] = df_clean['Diagnosis_Code'].fillna('Unknown')

print("Missing categorical values filled with 'Unknown'")
print("\nFinal categorical value counts:")
print("\nGender:")
print(df_clean['Gender'].value_counts())
print("\nEthnicity:")
print(df_clean['Ethnicity'].value_counts())
print("\nDiagnosis_Code:")
print(df_clean['Diagnosis_Code'].value_counts())
## Step 7: Normalize mixed units and engineer BMI feature

Healthcare data often contains mixed measurement units due to different countries or systems using different standards (metric vs imperial). You need to:

1. **Normalize Weight**: Convert all weights to kg (from mixed kg and lbs)
2. **Convert Height**: Convert cm to meters for BMI calculation
3. **Engineer BMI**: Body Mass Index is a clinically important derived feature

**BMI Formula**: BMI = Weight(kg) / Height(m)²

**BMI Categories**:
- Underweight: < 18.5
- Normal: 18.5 - 24.9
- Overweight: 25 - 29.9
- Obese: ≥ 30
# Function to convert weight to kg (handles both numeric kg and string 'lbs' format)
def weight_to_kg(x):
    """
    Convert weight to kilograms.
    Handles numeric values (assumed kg) and strings with 'lbs' suffix.

    Examples:
    70 -> 70.0 kg
    '150lbs' -> 68.04 kg
    '150 lbs' -> 68.04 kg
    """
    if pd.isna(x):
        return np.nan

    # If already numeric, assume it's in kg
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    # Handle string values
    s = str(x).strip().lower()

    # Check for lbs pattern (e.g., '150lbs' or '150 lbs')
    match = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*lbs?\s*$', s)
    if match:
        lbs = float(match.group(1))
        return round(lbs * 0.45359237, 2)  # Convert lbs to kg

    # Try to parse as numeric (assume kg)
    try:
        return float(s)
    except:
        return np.nan

# Apply weight conversion
df_clean['Weight_kg'] = df_clean['Weight'].apply(weight_to_kg)

print("Weight conversion examples:")
display(df_clean[['Weight', 'Weight_kg']].head(10))
# Convert height from cm to meters
df_clean['Height_cm'] = pd.to_numeric(df_clean['Height_cm'], errors='coerce')
df_clean['Height_m'] = df_clean['Height_cm'] / 100.0

print("Height converted from cm to meters")
# Calculate BMI (Body Mass Index)
df_clean['BMI'] = df_clean.apply(
    lambda row: round(row['Weight_kg'] / (row['Height_m'] ** 2), 2)
    if pd.notna(row['Weight_kg']) and pd.notna(row['Height_m']) and row['Height_m'] > 0
    else np.nan,
    axis=1
)

print("BMI calculated successfully")
print("\nSample of engineered features:")
display(df_clean[['Weight', 'Weight_kg', 'Height_cm', 'Height_m', 'BMI']].head(10))
## Step 8: Handle missing values with imputation

Missing values are inevitable in healthcare data. Common causes include:
- Tests not performed for all patients
- Data entry errors
- Equipment failures
- Patient privacy restrictions

### Why use Median instead of Mean?

For healthcare data, **median imputation** is often preferred over mean because:

1. **Robust to outliers**: Healthcare data often contains extreme values (very high glucose, unusual ages)
2. **Mean is sensitive**: A few extreme values can skew the mean significantly
3. **Median represents center**: The middle value of sorted data, unaffected by extremes
4. **Preserves distribution**: Better maintains the shape of skewed distributions
5. **Simple and fast**: Computationally efficient with no assumptions about distribution

You'll impute missing values in numeric columns (Age, Weight_kg, Height_cm, BMI, Glucose_mg_dL) using their respective medians.
# Check missing values before imputation
print("Missing values before imputation:")
numeric_cols = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Glucose_mg_dL']
print(df_clean[numeric_cols].isnull().sum())
# Median imputation for numeric columns
for col in numeric_cols:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
    print(f"Imputed {col} with median = {median_val}")
# Verify imputation
print("\nMissing values after imputation:")
print(df_clean[numeric_cols].isnull().sum())
print("\nAll numeric missing values successfully imputed!")
## Step 9: Parse dates and engineer temporal features

Temporal features can be highly informative in healthcare:
- **Diagnosis year**: May reflect changes in diagnostic practices or disease prevalence
- **Time since diagnosis**: Important for understanding disease progression
- **Seasonal patterns**: Some conditions vary by time of year

You'll parse the inconsistent date formats and extract useful temporal features.
# Parse dates with mixed formats (YYYY-MM-DD and DD/MM/YYYY)
df_clean['Diagnosis_Date_parsed'] = pd.to_datetime(
    df_clean['Diagnosis_Date'],
    errors='coerce',  # Convert unparseable dates to NaT (Not a Time)
    dayfirst=True     # Assume day comes first in ambiguous formats
)

print("Date parsing results:")
print(f"Successfully parsed: {df_clean['Diagnosis_Date_parsed'].notna().sum()} dates")
print(f"Failed to parse: {df_clean['Diagnosis_Date_parsed'].isna().sum()} dates")

print("\nSample of original vs parsed dates:")
display(df_clean[['Diagnosis_Date', 'Diagnosis_Date_parsed']].head(10))
# Extract year from diagnosis date
df_clean['Diagnosis_Year'] = df_clean['Diagnosis_Date_parsed'].dt.year

# Calculate days since diagnosis (relative to most recent date in dataset)
ref_date = df_clean['Diagnosis_Date_parsed'].max()
if pd.isna(ref_date):
    ref_date = pd.to_datetime("today")

df_clean['Days_Since_Diagnosis'] = (
    ref_date - df_clean['Diagnosis_Date_parsed']
).dt.days

print(f"\nReference date for calculating time since diagnosis: {ref_date.date()}")
print("\nSample of engineered temporal features:")
display(df_clean[['Diagnosis_Date_parsed', 'Diagnosis_Year', 'Days_Since_Diagnosis']].head(10))
## Step 10: Encode categorical variables

Machine learning algorithms require numeric input. Categorical variables must be converted to numbers through **encoding**.

### One-hot encoding

One-hot encoding creates **binary (0/1) columns** for each category:

**Example**: If Diagnosis_Code has values ['DEP', 'OCD', 'ANX']
- Creates columns: `Diagnosis_Code_DEP`, `Diagnosis_Code_OCD`, `Diagnosis_Code_ANX`
- A patient with 'OCD' gets: [0, 1, 0]

**Why use one-hot encoding?**
- Treats all categories equally (no implicit ordering)
- Works with all ML algorithms
- Prevents models from assuming numerical relationships between categories

**Alternative**: Label Encoding (1, 2, 3...) should only be used for ordinal data with natural ordering.

You'll apply one-hot encoding to Gender, Ethnicity, and Diagnosis_Code.
# Apply one-hot encoding to categorical columns
print("Columns before encoding:")
print(df_clean.columns.tolist())
print(f"Total columns: {len(df_clean.columns)}")

df_final = pd.get_dummies(
    df_clean,
    columns=['Diagnosis_Code', 'Gender', 'Ethnicity'],
    drop_first=False  # Keep all columns (set True to drop one for linear models)
)

print("\nColumns after encoding:")
print(df_final.columns.tolist())
print(f"Total columns: {len(df_final.columns)}")
print(f"\nNew encoded columns created: {len(df_final.columns) - len(df_clean.columns)}")
# Preview the encoded dataset
print("Sample of encoded data:")
display(df_final.head())
## Step 11: Scale numeric features

### Why scale features?

Many machine learning algorithms are sensitive to feature scale:
- **Example**: Age (range 0-100) vs Glucose (range 70-500)
- Without scaling, algorithms may give more importance to features with larger values
- Algorithms affected: Logistic Regression, SVM, KNN, Neural Networks, K-Means
- Algorithms NOT affected: Tree-based models (Decision Trees, Random Forest, XGBoost)

### StandardScaler (Z-score normalization)

**Formula**: z = (x - μ) / σ
- Transforms data to have **mean = 0** and **standard deviation = 1**
- **Best for**: Algorithms assuming normal distribution (Linear/Logistic Regression, SVM)
- **Range**: Typically between -3 and +3 (but unbounded)

You'll apply StandardScaler to all numeric features.
# Define numeric columns to scale
numeric_cols_to_scale = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Glucose_mg_dL', 'Days_Since_Diagnosis']

# Filter to only existing columns
numeric_cols_existing = [col for col in numeric_cols_to_scale if col in df_final.columns]

print(f"Scaling {len(numeric_cols_existing)} numeric features:")
print(numeric_cols_existing)
# Fill any remaining missing values with median before scaling
df_final[numeric_cols_existing] = df_final[numeric_cols_existing].fillna(
    df_final[numeric_cols_existing].median()
)

print("Verified no missing values before scaling:")
print(df_final[numeric_cols_existing].isna().sum())
# Apply StandardScaler
scaler = StandardScaler()
scaled_columns = [col + '_scaled' for col in numeric_cols_existing]
df_final[scaled_columns] = scaler.fit_transform(df_final[numeric_cols_existing])

print("Scaling complete!")
print("\nScaled features statistics (should have mean≈0, std≈1):")
display(df_final[scaled_columns].describe())
# Compare original vs scaled values
print("\nComparison of original vs scaled values:")
comparison_cols = ['Age', 'Age_scaled', 'Glucose_mg_dL', 'Glucose_mg_dL_scaled']
display(df_final[comparison_cols].head(10))
## Step 12: Save the cleaned dataset

Now that you've completed all preprocessing steps, you'll save the cleaned dataset to a CSV file. This file is now ready for:
- Exploratory data analysis (EDA)
- Machine learning model training
- Statistical analysis
- Sharing with team members (with PII removed)
# Save cleaned dataset
output_path = "healthcare_cleaned_data.csv"
df_final.to_csv(output_path, index=False)

print(f"✓ Cleaned dataset saved to: {output_path}")
print(f"\nFinal dataset shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
print(f"Original dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
# Display working directory
import os
print(f"\nFile saved in directory: {os.getcwd()}")
## Step 13: Evaluate data cleaning results

Let's compare the raw and cleaned datasets to verify preprocessing was successful.
# Compare column structures
print("="*60)
print("COLUMN COMPARISON")
print("="*60)
print(f"\nRaw data columns ({len(df.columns)}):")
print(df.columns.tolist())
print(f"\nCleaned data columns ({len(df_clean.columns)}):")
print(df_clean.columns.tolist())
print(f"\nFinal encoded data columns ({len(df_final.columns)}):")
print(df_final.columns.tolist())
# Compare missing values
print("\n" + "="*60)
print("MISSING VALUES COMPARISON")
print("="*60)
print("\nRaw data missing values:")
print(df.isna().sum())
print(f"\nTotal missing values in raw data: {df.isna().sum().sum()}")

print("\nCleaned data missing values:")
print(df_clean.isna().sum())
print(f"\nTotal missing values in cleaned data: {df_clean.isna().sum().sum()}")
# Compare statistical summaries
print("\n" + "="*60)
print("STATISTICAL SUMMARY COMPARISON")
print("="*60)
print("\nRaw data summary:")
display(df.describe())

print("\nCleaned data summary:")
display(df_clean.describe())
# Compare categorical standardization (Gender example)
print("\n" + "="*60)
print("CATEGORICAL STANDARDIZATION - GENDER EXAMPLE")
print("="*60)
print("\nRaw Gender value counts:")
print(df['Gender'].value_counts(dropna=False))

print("\nCleaned Gender value counts:")
print(df_clean['Gender'].value_counts(dropna=False))

print(" Successfully standardized from 6 variations to 4 consistent categories!")


# Exercises

Now it's your turn! Apply what you've learned to a new synthetic healthcare dataset. The following exercises will test your understanding of the data preprocessing pipeline.
## Exercise 1: Load and prepare data

Load the `synthetic_data.csv` file into a DataFrame and create a clean working copy.
# your code goes here
<details>
    <summary>Click here for a hint</summary>
    
Use the `read_csv()` function to load the data, then use `.copy()` to create a working copy. Reference **Step 1** for the exact syntax.

</details>
<details>
    <summary>Click here for solution</summary>

```python
# Load the synthetic healthcare data
df = pd.read_csv("https://foundations-of-healthcare-data-analytics-4e579d.gitlab.io/labs/Cleaning_and_Validating_Healthcare_Data_Using_Python/synthetic_data.csv")

# Create a working copy
df_clean = df.copy()

# Display column names to verify
print("Columns in dataset:")
print(df_clean.columns.tolist())
print(f"\nDataset loaded: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
```

</details>
## Exercise 2: Remove personal data

Identify and remove all PII (Personally Identifiable Information) columns from the dataset. Common PII includes: Patient_ID, Name, Address, Phone, Email.
# your code goes here
<details>
    <summary>Click here for a hint</summary>
    
Use the `.drop()` method with `columns` parameter. Set `errors='ignore'` to avoid errors if a column doesn't exist. Reference **Step 4** for the syntax.

</details>
<details>
    <summary>Click here for solution</summary>

```python
# Define PII columns to remove
pii_cols = ['Patient_ID', 'Name', 'Address', 'Phone', 'Email']

# Remove PII columns (only if they exist)
df_clean = df_clean.drop(
    columns=[col for col in pii_cols if col in df_clean.columns],
    errors='ignore'
)

print("After removing PII columns:")
print(df_clean.columns.tolist())
print(f"\nColumns remaining: {len(df_clean.columns)}")
```

</details>
## Exercise 3: Drop duplicate rows

Check for and remove any duplicate rows in the dataset. Report how many duplicates were found and removed.
# your code goes here
<details>
    <summary>Click here for a hint</summary>
    
Use `.drop_duplicates()` method with `keep='first'` parameter. Count rows before and after to see how many were removed. Reference **Step 5**.

</details>
<details>
    <summary>Click here for solution</summary>

```python
# Count rows before deduplication
rows_before = len(df_clean)

# Remove exact duplicate rows (keep first occurrence)
df_clean = df_clean.drop_duplicates(keep='first')

# Count rows after deduplication
rows_after = len(df_clean)

# Report results
print(f"Rows before deduplication: {rows_before}")
print(f"Rows after deduplication: {rows_after}")
print(f"Duplicate rows removed: {rows_before - rows_after}")
```

</details>
## Exercise 4: Handle missing values in numeric columns

Identify all numeric columns, check for missing values, and impute them using the median strategy. Verify that all missing values have been filled.
# your code goes here
<details>
    <summary>Click here for a hint</summary>
    
First, use `.select_dtypes(include=[np.number])` to get numeric columns. Then use `.median()` and `.fillna()` for each column. Reference **Step 8** for the complete approach.

</details>
<details>
    <summary>Click here for solution</summary>

```python
# Identify numeric columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns found: {numeric_cols}")

# Check missing values before imputation
print("\nMissing values before imputation:")
print(df_clean[numeric_cols].isnull().sum())

# Convert to numeric (coerce invalid values to NaN)
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Median imputation
for col in numeric_cols:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
    print(f"Imputed {col} with median = {median_val}")

# Verify
print("\nMissing values after imputation:")
print(df_clean[numeric_cols].isnull().sum())
print("\n✓ All numeric missing values imputed successfully!")
```

</details>
## Exercise 5: Standardize categorical variables

Standardize the inconsistent categorical entries in the `Gender` and `Disease_Type` columns. Print the unique values before and after standardization.

**Hint**: Gender variations might include: M/Male/m/F/Female/f  
**Hint**: Disease_Type variations might include: ckd/CKD, LD/ld/Liver Disease

<details>
    <summary>Click here for a hint</summary>
    
Use `.str.lower()` and `.str.strip()` first, then create a mapping dictionary to standardize variations. Reference **Step 6** for the complete pattern.

</details>
<details>
    <summary>Click here for solution</summary>

```python
# Standardize Gender
print("Before standardization - Gender:")
print(df_clean['Gender'].value_counts(dropna=False))

df_clean['Gender'] = df_clean['Gender'].astype(str).str.strip().str.lower()
gender_map = {
    'male': 'Male', 'm': 'Male',
    'female': 'Female', 'f': 'Female',
    'other': 'Other',
    'nan': np.nan, 'none': np.nan
}
df_clean['Gender'] = df_clean['Gender'].replace({'nan': np.nan})
df_clean['Gender'] = df_clean['Gender'].map(
    lambda x: gender_map.get(x, x.capitalize() if pd.notna(x) else x)
)

print("\nAfter standardization - Gender:")
print(df_clean['Gender'].value_counts(dropna=False))


```

</details>
---

# Congratulations!

You have successfully completed this lab on healthcare data preprocessing! You've learned how to systematically clean messy real-world data by handling missing values, removing duplicates, standardizing inconsistent entries, engineering meaningful features, and preparing data for machine learning. These skills are essential for any data science project, especially in healthcare where data quality directly impacts patient outcomes and model reliability.

## Authors

[Ramesh Sannareddy](https://www.linkedin.com/in/rsannareddy/)

Copyright © 2025 SkillUp. All rights reserved.
