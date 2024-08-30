import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Function to convert image to base64
def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def fixthecode(val,max,min):
    res = (val-min)/(max-min)
    return res

def Rfixthecode(val,max,min):
    res = ((val*(max-min)) + min)
    return res


file_path1 = r"C:\Users\DELL\Downloads\CARdeko\id.xlsx"
df1 = pd.read_excel(file_path1)




# Load the pre-trained model
model_path = r"C:\Users\DELL\Downloads\CARdeko\pic2.pkl"
model = joblib.load(model_path)

# Load the scaler
#scaler_path = r"C:\Users\DELL\Downloads\CARdeko\scaler2.pkl"
#scaler = joblib.load(scaler_path)

# Define the function for making predictions
def predict_price(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Convert image to base64 and use it in CSS
background_image_path = r"C:\Users\DELL\Downloads\CARdeko\cars1.png"
background_image_base64 = image_to_base64(background_image_path)



# Load the Excel file
file_path = r"C:\Users\DELL\Downloads\CARdeko\cleaned_cars_dataset.xlsx"
df = pd.read_excel(file_path)

# Calculate min and max values for each column

pmin = int(df['price'].min())
pmax = int(df['price'].max())

Mileage_min = int(df['Mileage'].min())
Mileage_max = int(df['Mileage'].max())

Kms_Driven_min = int(df['KmsDriven'].min())
Kms_Driven_max = int(df['KmsDriven'].max())

Model_Year_min = int(df['modelYear'].min())
Model_Year_max = int(df['modelYear'].max())

owner_no_min = int(df['ownerNo'].min())
owner_no_max = int(df['ownerNo'].max())

engine_displacement_min = int(df['EngineDisplacement'].min())
engine_displacement_max = int(df['EngineDisplacement'].max())

max_power_min = int(df['MaxPower'].min())
max_power_max = int(df['MaxPower'].max())

torque_min = int(df['Torque'].min())
torque_max = int(df['Torque'].max())

centralVariantId_min = int(df['centralVariantId'].min())
centralVariantId_max = int(df['centralVariantId'].max())


# Streamlit UI with background image and custom styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{background_image_base64});
        background-size: cover;
        background-position: center;
    }}
    h1, h2, h3, h4, h5 {{
        color: white;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }}
    .stButton button {{
        background-color: #FF5733;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
    }}
    .stButton button:hover {{
        background-color: #C70039;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 CARdeko - Car Price Prediction")
st.write("Enter the car details below to get an estimated price. 🚘")

# Collect user inputs
st.subheader("Car Specifications")

city_options = ['Bangalore', 'Kolkata', 'Jaipur', 'Hyderabad', 'Delhi', 'Chennai']
city = st.selectbox("City", city_options)
city_mapping = {city: idx for idx, city in enumerate(city_options)}
city_value = (city_mapping[city])
city_value = fixthecode(city_value,5,0)

# No of previous Owners
owner_no_options = [i for i in range(owner_no_min, owner_no_max + 1)]
owner_no = st.selectbox("No of previous Owners", owner_no_options)
owner_no = fixthecode(owner_no,owner_no_max,owner_no_min)

oem_options = [
    'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Datsun', 'Fiat', 'Ford', 'HindustanMotors',
    'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'LandRover', 'Lexus', 'Mahindra',
    'MahindraRenault', 'MahindraSsangyong', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi',
    'MG', 'Nissan', 'Opel', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
]
oem = st.selectbox("Manufacturer", oem_options)
oem_mapping = {oem: idx for idx, oem in enumerate(oem_options)}
oem_value = oem_mapping[oem]
oem_value = fixthecode(oem_value,len(oem_options)-1,0)

# Model Year
model_year_options = [i for i in range(Model_Year_min, Model_Year_max + 1)]
model_year = st.selectbox("Model Year", model_year_options)
model_year = fixthecode(model_year,Model_Year_max,Model_Year_min)

variant_column = df1['VariantName-VariantId']
selected_variant = st.selectbox('Variant Name-Variant Id:', variant_column)


# Central Variant ID
central_variant_id_options = [i for i in range(centralVariantId_min, centralVariantId_max + 1)]
centralVariantId = st.selectbox("Variant ID *(Select the Varient Id which is corresponding to the selected Varient Name)", central_variant_id_options)
centralVariantId = fixthecode(centralVariantId,centralVariantId_max,centralVariantId_min)

iv_options = ["ThirdPartyinsurance", 'Comprehensive', 'ZeroDep', 'NotAvailable']
iv = st.selectbox("Insurance Validity", iv_options)
iv_mapping = {iv: idx for idx, iv in enumerate(iv_options)}
iv_value = iv_mapping[iv]
iv_value = fixthecode(iv_value,3,0)

fuel_options = ['Petrol', 'Diesel', 'Electric', 'LPG']
fuel_type = st.selectbox("Fuel Type", fuel_options)
fuel_mapping = {fuel_type: idx for idx, fuel_type in enumerate(fuel_options)}
fuel_value = fuel_mapping[fuel_type]
fuel_value = fixthecode(fuel_value,3,0)

# Kms Driven
kms_driven_options = [i for i in range(Kms_Driven_min, Kms_Driven_max + 1,1000)]
kms_driven = st.selectbox("Kms Driven", kms_driven_options)
kms_driven = fixthecode(kms_driven,Kms_Driven_max,Kms_Driven_min)

# Engine Displacement
engine_displacement_options = [i for i in range(engine_displacement_min, engine_displacement_max + 1)]
engine_displacement = st.selectbox("Engine (cc)   *Select '0' for Electric car*", engine_displacement_options)
engine_displacement = fixthecode(engine_displacement,engine_displacement_max,engine_displacement_min)

# Mileage (kmpl)
mileage_options = [i for i in range(Mileage_min, Mileage_max + 1)]
mileage = st.selectbox("Mileage (kmpl)", mileage_options)
mileage = fixthecode(mileage,Mileage_max,Mileage_min)

# Max Power
max_power_options = [i for i in range(max_power_min, max_power_max + 1)]
max_power = st.selectbox("Max Power (bhp)", max_power_options)
max_power = fixthecode(max_power,max_power_max,max_power_min)

# Torque
torque_options = [i for i in range(torque_min, torque_max + 1)]
torque = st.selectbox("Torque (Nm)", torque_options)
torque = fixthecode(torque,torque_max,torque_min)


bt_options = ['sedan', 'SUV', "PickupTrucks", 'Minivans', 'MUV', 'Hybrids', 'Hatchback', 'Coupe']
bt = st.selectbox("Body Type", bt_options)
bt_mapping = {bt: idx for idx, bt in enumerate(bt_options)}
bt_value = bt_mapping[bt]
bt_value = fixthecode(bt_value,7,0)


# Prepare the input array for prediction
input_data = [ 
    city_value, owner_no,oem_value,model_year,centralVariantId,iv_value, 
    fuel_value,kms_driven, engine_displacement, mileage, max_power,torque, bt_value

]


# Prediction button
if st.button("🚀 Predict Price"):
    # Get the normalized predicted price
    normalized_predicted_price = predict_price(model, input_data)
    normalized_predicted_price = Rfixthecode(normalized_predicted_price,pmax,pmin)
    st.success(f"💰 The predicted car price is: ₹ {normalized_predicted_price:}")
