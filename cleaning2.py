

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
# Load the combined dataset
file_path = r"C:\Users\DELL\Downloads\CARdeko\cleaned_cars_dataset.xlsx"
df = pd.read_excel(file_path)

# Update the 'InsuranceValidity' column
df['InsuranceValidity'] = df['InsuranceValidity'].replace({
    'ThirdParty': 'ThirdPartyinsurance',
    '1': 'NotAvailable',
    '2': 'NotAvailable'
})
# 1. Handling 'InsuranceValidity' column
insurance_mapping = {
    'NotAvailable': 3,  
    'ThirdPartyinsurance': 0, 
    'Comprehensive': 1, 
    'ZeroDep': 2, 
}
df['InsuranceValidity'] = df['InsuranceValidity'].map(insurance_mapping)
df['InsuranceValidity'].fillna(0, inplace=True)

# 2. Handling 'FuelType' column
fuel_mapping = {
    'Diesel': 1, 
    'Electric': 2, 
    'LPG': 3, 
    'Petrol': 0, 
}
df['FuelType'] = df['FuelType'].map(fuel_mapping)
df['FuelType'].fillna(0, inplace=True)

# 3. Handling 'bt' column
bt_mapping = {
    'Sedan': 0, 
    'SUV': 1, 
    'PickupTrucks': 2, 
    'Minivans': 3, 
    'MUV': 4, 
    'Hybrids': 5,  
    'Hatchback': 6,
    'Coupe': 7
          
}
df['bt'] = df['bt'].map(bt_mapping)
df['bt'].fillna(0, inplace=True)


# 4. Encoding Categorical Variables

City_mapping = {
    'Banglore': 0, 
    'Kolkata': 1,
    'Jaipur': 2,
    'Hyderabad': 3,
    'Delhi': 4,
    'Chennai': 5
          
}
df['City'] = df['City'].map(City_mapping)

 
#test
#transmission_mapping = {
    #'Manual': 0, 
    #'Automatic': 1
          
#}
#df['Transmission'] = df['Transmission'].map(transmission_mapping)
#df['Transmission'].fillna(0, inplace=True)
 

#
oem_mapping = {
    'Audi': 0,
    'BMW': 1,
    'Chevrolet': 2,
    'Citroen': 3,
    'Datsun': 4,
    'Fiat': 5,
    'Ford': 6,
    'HindustanMotors': 7,
    'Honda': 8,
    'Hyundai': 9,
    'Isuzu': 10,
    'Jaguar': 11,
    'Jeep': 12,
    'Kia': 13,
    'LandRover': 14,
    'Lexus': 15,
    'Mahindra': 16,
    'MahindraRenault': 17,
    'MahindraSsangyong': 18,
    'Maruti': 19,
    'Mercedes-Benz': 20,
    'Mini': 21,
    'Mitsubishi': 22,
    'MG': 23,
    'Nissan': 24,
    'Opel': 25,
    'Porsche': 26,
    'Renault': 27,
    'Skoda': 28,
    'Tata': 29,
    'Toyota': 30,
    'Volkswagen': 31,
    'Volvo': 32
}

df['oem'] = df['oem'].map(oem_mapping)
df['oem'].fillna(0, inplace=True)

from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

# Function to remove outliers based on IQR for given columns
def remove_outliers_iqr(df, columns):
    for col in columns:
        # Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the DataFrame to keep only values within the bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

# Load your dataset (assuming it's already a pandas DataFrame)
# df = pd.read_excel("path_to_your_file.xlsx")

# List of numerical columns for normalization and outlier removal
numerical_columns = ['modelYear', 'KmsDriven', 'price', 'Mileage', 'ownerNo',
                     'oem', 'City','centralVariantId', 'InsuranceValidity','FuelType' ,'EngineDisplacement', 'MaxPower', 'Torque', 'bt']



# Step 1: Normalize the  numerical columns
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 3: Remove outliers from the specified columns
df_cleaned = remove_outliers_iqr(df, numerical_columns)

# Step 4: Save the preprocessed dataset to a new Excel file
output_file_path = r"C:\Users\DELL\Downloads\CARdeko\preprocessed_2_cars_dataset.xlsx"
df_cleaned.to_excel(output_file_path, index=False)

print(f"Preprocessing completed and saved to {output_file_path}")












