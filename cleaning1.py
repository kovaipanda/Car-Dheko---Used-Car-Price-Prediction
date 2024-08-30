import pandas as pd
import re

# Load the combined dataset
file_path = r"C:\Users\DELL\Downloads\CARdeko\cars_dataset.xlsx"
df = pd.read_excel(file_path)


unique_cities = df['City'].unique()
print(unique_cities)

# Function to convert 'price' column
def convert_price(price_str):
    if pd.isna(price_str):
        return None
    price_str = price_str.replace('₹', '').replace(',', '').strip().lower()
    
    # Convert 'Lakh' to number
    if 'lakh' in price_str:  # case-insensitive check
        return float(price_str.replace('lakh', '').strip()) * 100000
    else:
        try:
            return float(price_str)
        except ValueError:
            return None  # If the value can't be converted to float

# Function to convert 'Mileage' column
def convert_mileage(mileage_str):
    if pd.isna(mileage_str):
        return None
    mileage_str = re.sub(r'[^0-9.]', '', mileage_str)  # Keep only numbers and periods
    return float(mileage_str) if mileage_str else None

# Function to convert 'KmsDriven' column
def convert_kms_driven(kms_str):
    if pd.isna(kms_str):
        return None
    # Remove non-numeric characters such as "Kms" or commas
    kms_str = re.sub(r'[^0-9]', '', kms_str)
    return int(kms_str)

# Apply the conversion functions
df['price'] = df['price'].apply(convert_price)
df['Mileage'] = df['Mileage'].apply(convert_mileage)
df['KmsDriven'] = df['KmsDriven'].apply(convert_kms_driven)


# Fill missing values with median for 'price', 'Mileage', and 'KmsDriven' columns
df['price'].fillna(df['price'].median(), inplace=True)
df['Mileage'].fillna(df['Mileage'].median(), inplace=True)
df['KmsDriven'].fillna(df['KmsDriven'].median(), inplace=True)

# Handle missing values in 'InsuranceValidity' using mode or new category
# Mode imputation
mode_insurance_validity = df['InsuranceValidity'].mode()[0]
df['InsuranceValidity'].fillna(mode_insurance_validity, inplace=True)

# OR, create a new category for missing values
# df['InsuranceValidity'].fillna('Unknown', inplace=True)


def convert_EngineDisplacement(EngineDisplacement_str):
    if pd.isna(EngineDisplacement_str):
        return None
    # Remove non-numeric characters such as "EngineDisplacement" or commas
    EngineDisplacement_str = re.sub(r'[^0-9]', '', EngineDisplacement_str)
    return int(EngineDisplacement_str)

# Function to extract numeric value (float or integer) from the MaxPower string
def extract_max_power_value(power_value):
    # Ensure the input is treated as a string
    if isinstance(power_value, (float, int)):
        power_value = str(power_value)
    elif not isinstance(power_value, str):
        return None
    
    # Regular expression to match numbers (with optional decimal point) followed by optional 'bhp' or 'BHP'
    match = re.search(r'(\d+\.?\d*)(?:\s*[bB][Hh][Pp])?', power_value)
    if match:
        return float(match.group(1))  # Return the extracted number as float
    return None  # Return None if no match

# Function to extract numeric value (float or integer) from the Torque string

def extract_torque_value(torque_value):
    # Ensure the input is treated as a string
    if isinstance(torque_value, (float, int)):
        torque_value = str(torque_value)
    elif not isinstance(torque_value, str):
        return None
    
    # Regular expression to match numbers (with optional decimal point) followed by optional 'Nm' or 'nm'
    match = re.search(r'(\d+\.?\d*)(?:\s*[Nn][Mm])?', torque_value)
    if match:
        return float(match.group(1))  # Return the extracted number as float
    return None  # Return None if no match




# Apply the conversion functions
df['EngineDisplacement'] = df['EngineDisplacement'].apply(convert_EngineDisplacement)
df['MaxPower'] = df['MaxPower'].apply(extract_max_power_value)
df['Torque'] = df['Torque'].apply(extract_torque_value)


# List of columns to delete
#it has no value
#ft is same as fuel type
#km same as km driven
#owner,Ownership same as owner no but in string
#priceActual,priceSaving,priceFixedText has less to no data
#trendingText_imgUrl,trendingText_heading =Treanding car,trendingText_desc =Highchancesofsaleinnext6days
#RegistrationYear,YearofManufacture same as ModelYear but in string
#model is same as oem but extended version
#variantName because we have centralVariantId
#WheelSize many data mising
#RTO is same as City like TN - Chennai
#Engine same as EngineDisplacement
#Transmission and Seats has no impact : found in eda

columns_to_delete = ['it', 'ft','km','owner','Ownership','priceActual','priceSaving','priceFixedText','trendingText_imgUrl','trendingText_heading','trendingText_desc','RegistrationYear','YearofManufacture','model','variantName','WheelSize','RTO','Engine','Seats','Transmission']

# Delete the columns
df.drop(columns=columns_to_delete, inplace=True)


# Save the cleaned dataset to a new Excel file
output_file_path = r"C:\Users\DELL\Downloads\CARdeko\cleaned_cars_dataset.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Data cleaned and saved successfully to {output_file_path}")
