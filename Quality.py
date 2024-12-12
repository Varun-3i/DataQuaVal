import pandas as pd
import numpy as np
import re
from ydata_profiling import ProfileReport

# Load dataset
def load_dataset(path):
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")
        df.columns = df.columns.str.strip()  # Strip column names
        return df
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")

# Preprocess columns
def preprocess_column(column, dtype):
    if dtype == "date":
        return pd.to_datetime(column, errors="coerce")
    elif dtype == "numeric":
        return pd.to_numeric(column.replace({',': ''}, regex=True), errors="coerce")  # Handle numbers with commas
    else:
        return column

# Preprocess entire dataset
def preprocess_dataset(df):
    for col in df.columns:
        if "date" in col.lower():
            df[col] = preprocess_column(df[col], "date")
        elif (
            df[col].dtype == "object" 
            and df[col].str.match(r"^-?\d+(\.\d+)?$", na=False).any()
        ):
            df[col] = preprocess_column(df[col], "numeric")
    return df

# Handle missing values
def handle_missing_values(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    return df

# Remove duplicates
def remove_duplicates(df, key_columns=None):
    if key_columns:
        duplicates = df[df.duplicated(subset=key_columns, keep=False)]
        if not duplicates.empty:
            print(f"Duplicate entries found in key columns {key_columns}:\n", duplicates)
    return df.drop_duplicates()

# Standardize text columns
def standardize_text_columns(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Calculate scores
def completeness_score(column):
    return (len(column) - column.isnull().sum()) / len(column) * 100

def uniqueness_score(column):
    return column.nunique() / len(column) * 100

def validity_score(column, validation_function=None):
    if validation_function:
        valid_entries = column.apply(validation_function).sum()
        return valid_entries / len(column) * 100
    return 100

def timeliness_score(column, threshold_date):
    if pd.api.types.is_datetime64_any_dtype(column):
        if threshold_date is None:
            raise ValueError("Threshold date must be provided and cannot be None.")
        
        threshold_date = pd.to_datetime(threshold_date).tz_localize(None) 
        column = column.dt.tz_localize(None)
        
        timely_entries = column.apply(lambda x: pd.isna(x) or x >= threshold_date).sum()
        return timely_entries / len(column) * 100
    return 100

def accuracy_score(column, reference_column=None, threshold=None):
    if reference_column is None:
        return 100  # Assume 100% accuracy if no reference is provided
    if pd.api.types.is_numeric_dtype(column):
        if threshold is None:
            raise ValueError("Threshold must be provided for numerical columns.")
        correct_entries = (abs(column - reference_column) <= threshold).sum()
    else:
        correct_entries = (column == reference_column).sum()
    return correct_entries / len(column) * 100

def consistency_score(df, column1, column2):
    if column1 in df.columns and column2 in df.columns:
        inconsistent_entries = (df[column1] > df[column2]).sum()  
        return (1 - inconsistent_entries / len(df)) * 100
    return 100

def reliability_score(column):
    if pd.api.types.is_numeric_dtype(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        reliable_entries = ((column >= lower_bound) & (column <= upper_bound)).sum()
        return reliable_entries / len(column) * 100
    return 100

def calculate_scores(df, threshold_date=None):
    if threshold_date is None:
        threshold_date = pd.to_datetime("today")
    
    detailed_scores = {}
    
    for col in df.columns:
        column_data = df[col]
        column_scores = {
            "Completeness": completeness_score(column_data),
            "Uniqueness": uniqueness_score(column_data),
            "Validity": validity_score(
                column_data, 
                lambda x: bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", str(x)))
            ) if "email" in col.lower() else 100,
            "Timeliness": timeliness_score(column_data, threshold_date) if pd.api.types.is_datetime64_any_dtype(column_data) else 100,
            "Consistency": consistency_score(df, col, "end_date"),
            "Accuracy": 100, 
            "Reliability": reliability_score(column_data)
        }
        detailed_scores[col] = column_scores

    return pd.DataFrame(detailed_scores).T

def overall_quality_score(scores_df):
    return scores_df.mean().mean()

# Generate Detailed Data Quality Report
def generate_flagged_report(df, flagged_entries, output_path="flagged_data_report.html"):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<html><head><title>Flagged Data Report</title></head><body>")
            f.write("<h1>Flagged Data Report</h1>")
            for col, entries in flagged_entries.items():
                f.write(f"<h2>{col}</h2><pre>{entries}</pre>")
            f.write("</body></html>")
    except Exception as e:
        print(f"Error generating flagged data report: {e}")

# Main
if __name__ == "__main__":
    dataset_path = "Amazon_Reviews.csv"  
    df = load_dataset(dataset_path)
    df = preprocess_dataset(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df, key_columns=["OrderID"])  
    detailed_scores_df = calculate_scores(df)
    overall_score = overall_quality_score(detailed_scores_df)
    flagged_entries = {col: df[df[col].isnull()] for col in df.columns if df[col].isnull().any()}
    generate_flagged_report(df, flagged_entries)  
    print(f"Overall Data Quality Score: {overall_score:.2f}")
