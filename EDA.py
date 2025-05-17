import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import zscore, normaltest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import warnings
warnings.filterwarnings("ignore")
#lOAD DATASET
def load_data(filepath):
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext == '.xlsx':
            df = pd.read_excel(filepath ,engine='openpyxl')
        else:
            raise ValueError("Unsupported file format! Use CSV/Excel File Format.")

        print(f"Data loaded successfully from {filepath}\n")
        return df
    except Exception as e:
        print(f"File loading error: {e}")
        return None


# Basic Info
def basic_info(df):
    print("\nFirst 5 Rows : \n", df.head())
    print("\nLast 5 Rows :\n", df.tail())

    print("\nDataset Shape :\n", df.shape)

    print("\nColumn Data Types :\n", df.dtypes)

    print("\nDataset Size :\n", df.size)

    print("\nDataset Columns :\n", df.columns.tolist(), '\n')

    print("\nDuplicate Valus :\n", df.duplicated().sum())

    print("\nUnique Values per Column:\n")
    for col in df.columns:
        print(f"➙{col}: {df[col].nunique()} unique values")

    print("\nDataset Info Summary :\n")
    print(df.info())
# Separate categorical columns and numeric columns
def separate_columns (df):
    categorical_columns = df.select_dtypes ( include = ['object','category']).columns.tolist()
    numeric_columns = df.select_dtypes ( include = ['int64' , 'float64']).columns.tolist()

    print("\nCategorical Columns :")
    for columns in categorical_columns :
        print (f"{columns} ({df[columns].dtype}")
    print("\nNumerical Columns :")
    for columns in numeric_columns:
        print(f"{columns} ({df[columns].dtype})")
#Check for the missing values
def show_missing_values(df):
    print("\nMissing Values: \n", df.isnull().sum())
    missing_values_percent = (df.isnull().sum() / len(df)) * 100
    print("\nMissing Value Percentage:\n", missing_values_percent[missing_values_percent > 0], "\n")

    missing_values_df = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Missing Values Percent (%)': missing_values_percent
    })
    missing_values_df = missing_values_df[missing_values_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)

    if not missing_values_df.empty:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        #  # Missing Value Count (Barplot)
        sns.barplot(x=missing_values_df.index, y=missing_values_df['Missing Values'], palette="coolwarm", ax=ax[0])
        ax[0].set_xlabel('Missing Values Columns', fontsize=11)
        ax[0].set_ylabel('Missing Values', fontsize=11)
        ax[0].tick_params(axis='x', labelsize=10)
        ax[0].tick_params(axis='y', labelsize=10)
        ax[0].set_title('The Missing Values in the Dataset (Bar Chart)', fontsize=13)

         # Missing Value Count (Heatmap)
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax[1])
        ax[1].set_title('The Missing Values in the Dataset (HeatMap)', fontsize=13)
        ax[1].set_xlabel('Missing Values Columns', fontsize=11)
        ax[1].set_ylabel('Rows', fontsize=11)
        ax[1].tick_params(axis='x', labelsize=10)
        ax[1].tick_params(axis='y', labelsize=10)

        plt.tight_layout()
        plt.show()
    else:
        print("No Missing Value present in the Dataset.")


# Function to display basic statistics and visualizations about the dataset
def show_descriptive_stats(df):
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.empty:
        print("No numeric columns found.")

    print("\nFull Descriptive Statistics (Including All Columns)")
    print(df.describe(include='all'))

    summary_data = []

    # Median, Mode, IQR
    for col in numeric_df.columns:
        median = numeric_df[col].median()
        mode = numeric_df[col].mode()[0] if not numeric_df[col].mode().empty else "N/A"
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1

        summary_data.append({
            'Column': col,
            'Median': median,
            'Mode': mode,
            'IQR': iqr
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nNumeric Summary (Median, Mode, IQR) ")
    print(summary_df)

    # Unique Value Count per Column
    unique_counts = numeric_df.nunique().reset_index()
    unique_counts.columns = ['Column', 'Unique Values']
    unique_value_df = pd.DataFrame(unique_counts)
    print(unique_value_df)


# method to clean the dataset
def clean_data(df, method='fill'):
    # method = "fill" (default): fills missing values
    # method = "drop": drops missing values

    has_missing = df.isnull().sum().sum() > 0

    if not has_missing:
        print("No missing values found — skipping cleaning step.")
        return df

    df_clean = df.copy()

    if method == "drop":
        df_clean.dropna(inplace=True)
        print("All missing values dropped.")

    elif method == "fill":
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype == 'object' or df_clean[col].nunique() <= 30:
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                    df_clean[col].fillna(fill_val, inplace=True)
                else:
                    df_clean[col].fillna(method='ffill', inplace=True)

        print("Missing values filled (Categorical: Mode, Numerical: forward fill).")
    else:
        print("Invalid method. Use 'fill' or 'drop'.")

    before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    after = len(df_clean)
    print(f" Dropped {before - after} duplicate rows.")

    return df_clean


# Outlier Handling
def remove_outliers(df, z_thresh=3):
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include='number').columns
    total_removed_rows = 0
    outliers_found = False

    for col in numeric_cols:
        col_data = df_out[col].dropna()
        if len(col_data) < 8:
            print(f"{col}: Not enough data for normality test.")
            continue

        stat, p = normaltest(col_data)
        is_normal = p > 0.05

        before_rows = df_out.shape[0]

        if is_normal:
            z_scores = zscore(df_out[col])
            condition = np.abs(z_scores) < z_thresh
            if not condition.all():
                outliers_found = True
                df_out = df_out[condition]
                print(f"{col}: Normal distribution → Z-score used")

                # Show boxplot before removing
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=df_out[col])
                plt.title(f'Before Removing Outliers in {col} (Z-Score)')
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()

            else:
                print(f"{col}: No outliers found")
        else:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            condition = (df_out[col] >= lower) & (df_out[col] <= upper)
            if not condition.all():
                outliers_found = True
                df_out = df_out[condition]
                print(f"{col}: Non-normal distribution → IQR used")

                # Show boxplot before removing
                plt.figure(figsize=(6, 4))
                sns.boxplot(y=df_out[col])
                plt.title(f'Before Removing Outliers in {col} (IQR Method)')
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()
            else:
                print(f"{col}: No outliers found")

        after_rows = df_out.shape[0]
        removed = before_rows - after_rows
        total_removed_rows += removed
        if removed > 0:
            print(f"{col}: Removed {removed} rows")

    if not outliers_found:
        print("\nNo outliers found in any numeric columns.")
    else:
        print(f"\nTotal outliers removed: {total_removed_rows}")

    return df_out


# categorical analysis
def categorical_analysis(df, max_categories=20):
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"\nColumn: {col}")
        print(f"Unique Values: {unique_vals}")
        print(df[col].value_counts())

        if unique_vals > max_categories:
            print(f"Skipped plotting '{col}' (too many categories)\n")
            continue

        value_counts = df[col].value_counts()

        plt.figure(figsize=(18, 5))

        # Count Plot
        plt.subplot(1, 3, 1)
        sns.countplot(x=col, data=df, order=value_counts.index)
        plt.xticks(rotation=45)
        plt.title("Count Plot")

        # Bar Chart (Vertical)
        plt.subplot(1, 3, 2)
        value_counts.plot(kind='bar', color='skyblue')
        plt.xticks(rotation=45)
        plt.title("Bar Chart")

        # Horizontal Bar Chart
        plt.subplot(1, 3, 3)
        value_counts.plot(kind='barh', color='orange', edgecolor='black')
        plt.title("Horizontal Bar")

        plt.tight_layout()
        plt.show()

        # Swarm Plot for each numeric column against the categorical column
        for num_col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.swarmplot(x=col, y=num_col, data=df)
            plt.title(f"Swarm Plot: {col} vs {num_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
# Numerical columns

def numeric_analysis(df):
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in num_cols:
        print(f"\nColumn: {col}")
        print(f"Summary Statistics:\n{df[col].describe()}\n")

        plt.figure(figsize=(18, 5))

        # Histogram
        plt.subplot(1, 4, 1)
        sns.histplot(df[col], kde=False, bins=30, color='skyblue')
        plt.title(f"Histogram: {col}")

        # Boxplot
        plt.subplot(1, 4, 2)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"Boxplot: {col}")

        # KDE Plot
        plt.subplot(1, 4, 3)
        sns.kdeplot(df[col], color='red')
        plt.title(f"KDE Plot: {col}")

        # Distribution Plot (with KDE + Histogram)
        plt.subplot(1, 4, 4)
        sns.histplot(df[col], kde=True, color='orange')
        plt.title(f"Dist Plot: {col}")

        plt.tight_layout()
        plt.show()
# Correlation matrix

def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    corr = numeric_df.corr(method='pearson')

    print("\nPearson Correlation Matrix:\n")
    print(corr)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True, cbar=True)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# feature engineering
def feature_engineering_helper(df):
    print("Feature Engineering Suggestions:\n")

    # Date Features
    date_cols = df.select_dtypes(include=['datetime64[ns]', 'object']).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"\nDate Features from: {col}")
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            print(f"Created: {col}_year, {col}_month, {col}_day")
        except Exception as e:
            print(f"Skipped {col}: {e}")

    text_cols = df.select_dtypes(include='object').columns
    for col in text_cols:
        if df[col].apply(lambda x: isinstance(x, str)).all():
            df[f'{col}_length'] = df[col].apply(len)
            print(f"\nText Length Feature Created: {col}_length")

    print("\nEncoding Suggestions:")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals <= 10:
            print(f"{col}: Suggested → One-Hot Encoding")
        else:
            print(f"{col}: Suggested → Label Encoding ")

    return df


# Encoding
def apply_encoding(df, method='auto', threshold=10):
    print("\nEncoding columns based on unique value count")

    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        unique_vals = df[col].nunique()

        if unique_vals <= threshold:
            print(f"{col}: One-Hot Encoding applied")
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            print(f"{col}: Label Encoding applied")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def run_full_auto_eda(file_path, missing_method='fill', remove_outliers_flag=True, apply_encoding_flag=True):
    df = load_data(file_path)
    if df is None:
        print("Failed to load dataset")
        return

    print("\nBASIC INFO")
    basic_info(df)

    print("\nSeparate Columns ( Categorical And Numerical Columns)")
    separate_columns(df)

    print("\nMISSING VALUES")
    show_missing_values(df)

    print("\nDESCRIPTIVE STATS")
    show_descriptive_stats(df)

    print(f"\nDATA CLEANING using method: {missing_method}")
    df_cleaned = clean_data(df, method=missing_method)

    if remove_outliers_flag:
        print("\nOUTLIER REMOVAL")
        df_cleaned = remove_outliers(df_cleaned)
    else:
        print("\nOutlier removal skipped (as per user input)")

    print("\nCATEGORICAL COLUMNS ANALYSIS")
    categorical_analysis(df)

    print("\nNUMERIC COLUMNS ANALYSIS ")
    numeric_analysis(df)

    print("\nCORRELATION MATRIX")
    correlation_matrix(df)

    print("\nFEATURE ENGINEERING")
    df_cleaned = feature_engineering_helper(df_cleaned)
    if df_cleaned is None:
        print("Feature engineering failed.")
        return None

    if apply_encoding_flag:
        print("\nAPPLYING ENCODING")
        df_cleaned = apply_encoding(df_cleaned)
        if df_cleaned is None:
            print("Encoding failed. Skipping remaining steps.")
            return None
    else:
        print("\nEncoding skipped (as per user input)")

    print("\nEDA COMPLETED")
    print("\nFinal dataset First 5 Rows : \n", df.head())
    print("Final dataset shape:", df_cleaned.shape)

    return df_cleaned
