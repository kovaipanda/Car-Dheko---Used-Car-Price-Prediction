
import pandas as pd

# File paths for all datasets
file_paths = [
    r"C:\Users\DELL\Downloads\CARdeko\Banglore.xlsx",
    r"C:\Users\DELL\Downloads\CARdeko\Kolkata.xlsx",
    r"C:\Users\DELL\Downloads\CARdeko\Jaipur.xlsx",
    r"C:\Users\DELL\Downloads\CARdeko\Hyderabad.xlsx",
    r"C:\Users\DELL\Downloads\CARdeko\Delhi.xlsx",
    r"C:\Users\DELL\Downloads\CARdeko\Chennai.xlsx"
]

# Columns to keep from each file
columns_to_keep = ['City', 'it','ft','km',	'ownerNo',	'owner','oem',	'model'	,'modelYear'	,'centralVariantId'	,'variantName',	'price',	'priceActual',	'priceSaving',	'priceFixedText',	'trendingText_imgUrl',	'trendingText_heading',	'trendingText_desc'	,'RegistrationYear',	'InsuranceValidity',	'FuelType',	'Seats'	,'KmsDriven'	,'RTO',	'Ownership'	,'EngineDisplacement',	'Transmission',	'YearofManufacture'	,'Mileage'	,'Engine',	'MaxPower'	,'Torque'	,'WheelSize', 'bt']

# Initialize an empty list to store each DataFrame
dfs = []

# Iterate over the file paths, loading each dataset and keeping only the necessary columns
for file_path in file_paths:
    df = pd.read_excel(file_path)
    # Filter the DataFrame to only the specified columns
    df_filtered = df[columns_to_keep]
    # Append the filtered DataFrame to the list
    dfs.append(df_filtered)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataset to a new Excel file
output_file_path = r"C:\Users\DELL\Downloads\CARdeko\cars_dataset.xlsx"
combined_df.to_excel(output_file_path, index=False)

print(f"All datasets concatenated and saved successfully to {output_file_path}")






