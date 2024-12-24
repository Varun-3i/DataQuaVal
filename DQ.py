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
        return pd.to_numeric(column, errors="coerce")
    else:
        return column

# Preprocess entire dataset
def preprocess_dataset(df, date_columns=None, numeric_columns=None, text_columns=None, 
                        date_formats=None, categorical_columns=None, outlier_method=None):
    """
    Preprocesses a DataFrame by detecting and handling date, numeric, and text columns.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        date_columns (list, optional): List of columns to treat as dates. Defaults to None.
        numeric_columns (list, optional): List of columns to treat as numeric. Defaults to None.
        text_columns (list, optional): List of columns to treat as text. Defaults to None.
        date_formats (dict, optional): A dictionary mapping date columns to their formats. Defaults to None.
        categorical_columns (list, optional): List of columns to treat as categorical. Defaults to None.
        outlier_method (str, optional): Method for handling outliers in numeric columns. 
                                        Options: 'cap' (cap at percentiles), 'winsorize'. Defaults to None.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    date_columns = date_columns or []
    numeric_columns = numeric_columns or []
    text_columns = text_columns or []
    date_formats = date_formats or {}
    categorical_columns = categorical_columns or []

    # Process date columns
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], format=date_formats.get(col), errors="coerce")
            except Exception as e:
                print(f"Error converting column '{col}' to datetime: {e}")

    # Process numeric columns
    for col in numeric_columns:
        if col in df.columns:
            try:
                # Clean non-numeric characters and convert
                df[col] = df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Handle outliers
                if outlier_method == 'cap':
                    upper_lim = df[col].quantile(0.95)
                    lower_lim = df[col].quantile(0.05)
                    df[col] = np.clip(df[col], lower_lim, upper_lim)
                elif outlier_method == 'winsorize':
                    from scipy.stats.mstats import winsorize
                    df[col] = winsorize(df[col], limits=[0.05, 0.05])  # Cap at 5th and 95th percentiles

            except Exception as e:
                print(f"Error converting column '{col}' to numeric: {e}")

    # Process text columns
    for col in text_columns:
        if col in df.columns:
            try:
                # Trim whitespace and convert to lowercase
                df[col] = df[col].astype(str).str.strip().str.lower()
            except Exception as e:
                print(f"Error processing text column '{col}': {e}")

    # Process categorical columns
    for col in categorical_columns:
        if col in df.columns:
            try:
                # Example: Group infrequent categories
                top_categories = df[col].value_counts().head(10).index
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
            except Exception as e:
                print(f"Error processing categorical column '{col}': {e}")

    return df

# Handle missing values
#def handle_missing_values(df):
    #for col in df.select_dtypes(include=[np.number]).columns:
     #   df[col] = df[col].fillna(df[col].median())
#    for col in df.select_dtypes(include=["object"]).columns:
#        df[col] = df[col].fillna(df[col].mode()[0])
#    return df

# Remove duplicates
#def remove_duplicates(df):
    #return df.drop_duplicates()

# Standardize text columns
#def standardize_text_columns(df):
    #for col in df.select_dtypes(include=["object"]).columns:
        #df[col] = df[col].str.strip().str.lower()
    #return df

# Calculate scores

def completeness_score(column):
    """Calculate the completeness score of a column.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        float: Completeness score as a percentage.
    """
    if len(column) == 0:
        return 0.0  # Return 0% if the column is empty
    return (len(column) - column.isnull().sum()) / len(column) * 100

def uniqueness_score(column):
    """Calculate the uniqueness score of a column.

    Args:
        column (pd.Series): The column to evaluate.

    Returns:
        float: Uniqueness score as a percentage.
    """
    if len(column) == 0:
        return 0.0  # Return 0% if the column is empty
    return column.nunique() / len(column) * 100

def validity_score(column, validation_function=None):
    """Calculate the validity score of a column based on a validation function.

    Args:
        column (pd.Series): The column to evaluate.
        validation_function (callable, optional): A function to validate entries.

    Returns:
        float: Validity score as a percentage.
    """
    if len(column) == 0:
        return 0.0  # Return 0% if the column is empty
    if validation_function:
        valid_entries = column.apply(validation_function).sum()
        return valid_entries / len(column) * 100

def timeliness_score(column, threshold_date):
    """Calculate the timeliness score of a datetime column.

    Args:
        column (pd.Series): The datetime column to evaluate.
        threshold_date (pd.Timestamp): The date to compare against.

    Returns:
        float: Timeliness score as a percentage.

    Raises:
        ValueError: If threshold_date is None.
    """
    if pd.api.types.is_datetime64_any_dtype(column):
        if threshold_date is None:
            raise ValueError("Threshold date must be provided and cannot be None.")

        threshold_date = threshold_date.tz_localize(None)

        timely_entries = column.apply(lambda x: pd.isna(x) or x >= threshold_date).sum()
        return timely_entries / len(column) * 100

def accuracy_score(column, reference_column=None, threshold=None):
    """Calculate the accuracy score of a column compared to a reference column.

    Args:
        column (pd.Series): The column to evaluate.
        reference_column (pd.Series, optional): The reference column for comparison.
        threshold (float, optional): The allowable deviation for numerical comparisons.

    Returns:
        float: Accuracy score as a percentage.

    Raises:
        ValueError: If reference_column is None or threshold is None for numerical columns.
    """
    if reference_column is None:
        return 100.0  # Assume 100% accuracy if no reference is provided
    if len(column) == 0 or len(reference_column) == 0:
        return 0.0  # Return 0% if either column is empty

    if pd.api.types.is_numeric_dtype(column):
        if threshold is None:
            raise ValueError("Threshold must be provided for numerical columns.")

        correct_entries = (abs(column - reference_column) <= threshold).sum()
    else:
        correct_entries = (column == reference_column).sum()

    return correct_entries / len(column) * 100

def consistency_score(df, column1, column2=None, consistency_rule=None, default_score=100):
    """
    Calculates the consistency score by comparing two columns or applying a custom rule.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        column1 (str): The first column to compare.
        column2 (str, optional): The second column to compare. Defaults to None.
        consistency_rule (callable, optional): A custom consistency rule. 
                                               Should return True for consistent rows.
        default_score (float): Default score to return if no comparison is made. Defaults to 100.

    Returns:
        float: The consistency score as a percentage (0-100).
    """
    if column1 not in df.columns:
        print(f"Warning: Column '{column1}' does not exist. Returning default score.")
        return default_score

    if column2 and column2 not in df.columns:
        print(f"Warning: Column '{column2}' does not exist. Assuming 100% consistency.")
        return default_score

    if consistency_rule:
        inconsistent_entries = df.apply(consistency_rule, axis=1).sum()
    elif column2:
        if pd.api.types.is_numeric_dtype(df[column1]) and pd.api.types.is_numeric_dtype(df[column2]):
            inconsistent_entries = (df[column1] > df[column2]).sum()
        elif pd.api.types.is_datetime64_any_dtype(df[column1]) and pd.api.types.is_datetime64_any_dtype(df[column2]):
            inconsistent_entries = (df[column1] > df[column2]).sum()
        else:
            print(f"Warning: Columns '{column1}' and '{column2}' are not comparable. Returning default score.")
            return default_score
    else:
        # If no second column and no rule, assume consistency
        return default_score

    total_entries = len(df)
    consistency = (1 - inconsistent_entries / total_entries) * 100 if total_entries > 0 else default_score

    return round(consistency, 2)

def reliability_score(column):
    """Calculate the reliability score of a numerical column based on outlier detection.

    Args:
        column (pd.Series): The numeric column to evaluate.

    Returns:
        float: Reliability score as a percentage.

    Notes:
        Returns 100% for non-numeric columns or empty columns.
    """
    if pd.api.types.is_numeric_dtype(column):
        if len(column) == 0:
            return 0.0  # Return 0% if the column is empty

        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        reliable_entries = ((column >= lower_bound) & (column <= upper_bound)).sum()
        return reliable_entries / len(column) * 100
    
    return 100.0  # Return 100% reliability if the column is not numerical

def calculate_scores(df, threshold_date=None, reference_columns=None):
    """
    Calculates data quality scores for each column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        threshold_date (pd.Timestamp, optional): Threshold date for timeliness. Defaults to None (today).
        reference_columns (dict, optional): A dictionary mapping column names to their 
                                            reference columns for accuracy calculation. 
                                            Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with data quality scores for each column.
    """

    if threshold_date is None:
        threshold_date = pd.to_datetime("today")

    detailed_scores = {}

    for col in df.columns:
        column_data = df[col]

        # Calculate scores for each column
        column_scores = {
            "Completeness": completeness_score(column_data),
            "Uniqueness": uniqueness_score(column_data),
            "Validity": validity_score(
                column_data,
                lambda x: bool(
                    re.match(
                        r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", str(x)
                    )
                ),
            ) if "email" in col.lower() else 100,
            "Timeliness": timeliness_score(column_data, threshold_date) if pd.api.types.is_datetime64_any_dtype(column_data) else 100,
            "Consistency": consistency_score(df, col),  # You might need to adjust this based on your consistency logic
            "Accuracy": accuracy_score(column_data, reference_columns.get(col)) if reference_columns else 100,  # Use reference column if provided
            "Reliability": reliability_score(column_data)
        }

        detailed_scores[col] = column_scores

    scores_df = pd.DataFrame(detailed_scores).T
    return scores_df

def overall_quality_score(scores_df):
    return scores_df.mean().mean()

# Generate Detailed Data Quality Scores Report
def generate_detailed_report(df, detailed_scores_df, overall_score, output_path="detailed_data_quality_report.html", css_path="style.css"):
    try:
        html_content = [
            f"<html><head><title>Detailed Data Quality Report</title><link rel='stylesheet' href='{css_path}'></head><body>"
        ]

        html_content.append("<h1>Detailed Data Quality Scores</h1>")
        html_content.append(f"<h2>Overall Data Quality Score: {overall_score:.2f}</h2>")

        html_content.append("<table>")
        html_content.append("<tr><th>Column</th><th>Completeness</th><th>Uniqueness</th><th>Validity</th><th>Timeliness</th><th>Consistency</th><th>Accuracy</th><th>Reliability</th></tr>")

        for col, scores in detailed_scores_df.iterrows():
            html_content.append(
                f"<tr><td>{col}</td><td>{scores['Completeness']:.2f}%</td><td>{scores['Uniqueness']:.2f}%</td>"
                f"<td>{scores['Validity']:.2f}%</td><td>{scores['Timeliness']:.2f}%</td><td>{scores['Consistency']:.2f}%</td>"
                f"<td>{scores['Accuracy']:.2f}%</td><td>{scores['Reliability']:.2f}%</td></tr>"
            )

        html_content.append("</table></body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))

        print(f"Detailed Data Quality Report generated successfully: {output_path}")

    except Exception as e:
        print(f"Error generating detailed report: {e}")

# Generate Quality Summary Report
def generate_quality_summary(df, scores_df, output_path="quality_summary_report.html", css_path="summary_style.css"):
    try:
        html_content = [
            f"<html><head><title>Quality Summary Report</title><link rel='stylesheet' href='{css_path}'></head><body>"
        ]

        html_content.append("<h1>Quality Summary Report</h1>")

        # Loop through each quality metric (column-wise)
        for metric in scores_df.columns:
            # Identify columns that meet the passing threshold for the metric (80% or higher)
            columns_passing = scores_df[scores_df[metric] >= 80].index.tolist()
            passing_percentage = (len(columns_passing) / len(scores_df)) * 100
            html_content.append(f"<h2>{metric} ({passing_percentage:.2f}% Passing)</h2>")

            # List columns passing the threshold
            if columns_passing:
                html_content.append("<ul>")
                for col in columns_passing:
                    html_content.append(f"<li>{col}</li>")
                html_content.append("</ul>")
            else:
                html_content.append("<p>No columns passed this metric.</p>")

        html_content.append("</body></html>")

        # Write the content to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))

        print(f"Quality Summary Report generated successfully: {output_path}")

    except Exception as e:
        print(f"Error generating quality summary report: {e}")

# Generate YData Profiling Report
def generate_ydata_profiling_report(df, output_path="ydata_profiling_report.html"):
    try:
        profile = ProfileReport(df, title="YData Profiling Report", explorative=True)
        profile.to_file(output_path)
        print(f"YData Profiling Report generated successfully: {output_path}")
    except Exception as e:
        print(f"Error generating YData profiling report: {e}")

# Main block to run the data pipeline and generate reports
if __name__ == "__main__":
    dataset_path = "dataset_with_issues.csv"  # Update with your dataset path
    df = load_dataset(dataset_path)

    # Calculate data quality scores without cleaning the data
    detailed_scores_df = calculate_scores(df)
    overall_score = overall_quality_score(detailed_scores_df)

    # Generate detailed data quality report
    generate_detailed_report(df, detailed_scores_df, overall_score)

    # Generate quality summary report
    generate_quality_summary(df, detailed_scores_df)

    # Generate YData profiling report
    generate_ydata_profiling_report(df)