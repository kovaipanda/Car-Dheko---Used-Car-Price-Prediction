import pandas as pd
import json
import re

# Load the Excel file
file_path = r"C:\Users\DELL\Downloads\CARdeko\kolkata_cars.xlsx"
df = pd.read_excel(file_path, header=None)

# Function to handle and clean malformed dictionary strings
def clean_malformed_dict(dict_str):
    if not dict_str or pd.isna(dict_str):
        return None
    try:
        # Replace single quotes with double quotes for JSON compatibility
        dict_str = dict_str.replace("'", '"')

        # Remove unwanted characters and fix common issues
        dict_str = re.sub(r'(?<=\w)\s(?=\w)', '', dict_str)  # Remove spaces between words
        dict_str = dict_str.replace('None', 'null')  # Replace None with null
        dict_str = re.sub(r'(?<=\w)\s:\s(?=\w)', ':', dict_str)  # Fix spacing around colons

        # Convert string to JSON
        return json.loads(dict_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic string: {dict_str[:200]}...")  # Show first 200 characters for context
        return None
    
def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
            for i in v:
                items.extend(flatten_dict(i, f"{new_key}").items())
        else:
            items.append((new_key, v))
    return dict(items)

# Function to extract key-value pairs from the "top" list
def extract_key_value_pairs(data):
    key_value_dict = {}
    top_data = data.get('top', [])
    
    # Iterate through the list of key-value pairs
    for item in top_data:
        key = item.get('key')
        value = item.get('value')
        if key and value:
            key_value_dict[key] = value
    return key_value_dict

# Apply the parsing and extraction
data = []
for index, row in df.iterrows():
    for i in range(0,4):
        cell_value = row[i]
        if isinstance(cell_value, str):
            if i==0 :
                row_dict = clean_malformed_dict(cell_value)
                if row_dict:
                    flat_row = flatten_dict(row_dict)
                    data.append(flat_row)
            else:
                row_dict = clean_malformed_dict(cell_value)
                if row_dict:
                    extracted_data = extract_key_value_pairs(row_dict)
                    data.append(extracted_data)
        else:
            print(f"Non-string data in row {index}: {cell_value}")

# Create a DataFrame from the extracted key-value data
structured_df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(structured_df.head())

# Save the DataFrame to a new Excel file
output_file_path = r"C:\Users\DELL\Downloads\CARdeko\Kolkata.xlsx"
try:
    structured_df.to_excel(output_file_path, index=False)
    print(f"Data saved successfully to {output_file_path}")
except PermissionError as e:
    print(f"Permission error: {e}")
    print("Please ensure the file is not open in another program and that you have write permissions.")

#this step is repeated for all the cities

