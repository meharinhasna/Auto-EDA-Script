from EDA import run_full_auto_eda

file_path = input("Enter dataset path (e.g., titanic.csv): ")
missing_method = input("Missing value method ('fill' or 'drop'): ").strip().lower()
apply_feature_engineering_flag = input("Apply Feature Engineering? (yes/no): ").strip().lower() == "yes"
remove_outliers_flag = input("Remove outliers? (yes/no): ").strip().lower() == "yes"
apply_encoding_flag = input("Apply Encoding? (yes/no): ").strip().lower() == "yes"
df = run_full_auto_eda(
    file_path=file_path,
    missing_method=missing_method,
    remove_outliers_flag=remove_outliers_flag,
    apply_encoding_flag=apply_encoding_flag
)
# Saving the cleaned dataset
output_file = input("Enter filename to save cleaned dataset (e.g.,cleaned_data.csv): ")
df.to_csv(output_file, index=False)
print(f"Cleaned dataset saved to: {output_file}")