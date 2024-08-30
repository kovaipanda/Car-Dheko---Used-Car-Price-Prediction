Car-Dheko---Used-Car-Price-Prediction Scripts Overview
Here's a summary of the various scripts designed for processing, cleaning, analyzing, and modeling car price prediction data:

1. Car Extraction Script
Description: Processes and cleans Excel dataset (kolkata_cars.xlsx) containing malformed dictionary strings.
Steps:
Load data into a pandas DataFrame.
Clean malformed dictionary strings.
Flatten nested dictionaries.
Extract key-value pairs from 'top' field.
Save cleaned data to Kolkata.xlsx.
Requirements: Python 3.x, pandas, openpyxl.
Installation: pip install pandas openpyxl
Usage: Adjust file path and run the script.
2. Car Data Concatenation Script
Description: Consolidates car data from multiple city-specific Excel files into one dataset.
Steps:
Read data from city-specific files (Banglore.xlsx, Kolkata.xlsx, etc.).
Select relevant columns.
Concatenate data into a single DataFrame.
Save combined data to cars_dataset.xlsx.
Requirements: Python 3.x, pandas, openpyxl.
Installation: pip install pandas openpyxl
Usage: Ensure files are in the correct directory and run the script.
3. Car Data Cleaning Script
Description: Cleans the consolidated car dataset by standardizing key fields and handling missing values.
Steps:
Load data from cars_dataset.xlsx.
Clean and convert price, mileage, kilometers driven, engine specifications.
Handle missing values.
Remove unnecessary columns.
Save cleaned data to cleaned_cars_dataset.xlsx.
Requirements: Python 3.x, pandas, openpyxl, re.
Installation: pip install pandas openpyxl
Usage: Ensure dataset file is in the correct directory and run the script.
4. Car Cleaning2 Preprocess
Description: Further preprocesses the cleaned dataset by handling missing values, encoding categorical variables, normalizing numerical columns, and removing outliers.
Steps:
Load cleaned_cars_dataset.xlsx.
Handle missing values and encode categorical variables.
Normalize numerical columns using MinMaxScaler.
Remove outliers using IQR method.
Save preprocessed data to preprocessed_2_cars_dataset.xlsx.
Requirements: Python 3.x, pandas, numpy, scikit-learn, scipy, joblib.
Installation: pip install pandas numpy scikit-learn scipy joblib
Usage: Update file paths and run the script.
5. Eda1
Description: Performs exploratory data analysis (EDA) on the dataset, including scatter plots, histograms, box plots, and a correlation heatmap.
Steps:
Load dataset from preprocessed_2_cars_dataset.xlsx.
Display descriptive statistics.
Create scatter plots, histograms, box plots.
Generate a correlation heatmap.
Requirements: pandas, seaborn, matplotlib, openpyxl.
Installation: pip install pandas seaborn matplotlib openpyxl
Usage: Update file path and run the script.
6. Eda2
Description: Performs detailed analysis on car prices, including descriptive statistics, correlation analysis, and feature importance using RandomForestRegressor.
Steps:
Load dataset from preprocessed_2_cars_dataset.xlsx.
Calculate descriptive statistics.
Perform correlation analysis.
Assess feature importance using RandomForestRegressor.
Requirements: pandas, sklearn, numpy.
Installation: pip install pandas scikit-learn numpy openpyxl
Usage: Update file path and run the script.
7. Model
Description: Evaluates multiple machine learning models for car price prediction, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and K-Nearest Neighbors Regressor.
Steps:
Load dataset from preprocessed_2_cars_dataset.xlsx.
Split data into training and testing sets.
Train and evaluate models with and without hyperparameter tuning.
Compare model performance using metrics like MAE, MSE, and R2.
Requirements: pandas, scikit-learn, openpyxl.
Installation: pip install pandas scikit-learn openpyxl
Usage: Update file path and run the script.
8. Gradient Boosting
Description: Builds and evaluates a Gradient Boosting Regressor model for car price prediction, including hyperparameter tuning and saving/loading the model using pickle.
Steps:
Load dataset from preprocessed_2_cars_dataset.xlsx.
Prepare data for training.
Perform grid search for hyperparameter tuning.
Train and evaluate the Gradient Boosting model.
Save/load the model using pickle.
Requirements: pandas, numpy, scikit-learn, openpyxl, pickle.
Installation: pip install pandas numpy scikit-learn openpyxl
Usage: Update file path and run the script.
These scripts collectively cover the entire workflow from data extraction and cleaning to model evaluation and prediction for car price analysis. Adjust paths and column names as needed based on your dataset.
9.Streamlit
s_l.py: The main Streamlit application file.
id.xlsx: Excel file with variant names and IDs.
cleaned_cars_dataset.xlsx: Excel file with cleaned car data.
pic2.pkl: Pickle file containing the pre-trained model.
cars1.png: Background image used in the app.
image_to_base64(img_path): Converts an image file to a base64-encoded string for use in CSS.
fixthecode(val, max, min): Normalizes the input values for the model.
Rfixthecode(val, max, min): Rescales the predicted price back to the original range.
predict_price(model, input_data): Uses the pre-trained model to predict the car price based on user inputs.
Streamlit UI: Provides the user interface for input collection and displays the predicted price.

![image](https://github.com/user-attachments/assets/9dfa73a3-da20-48ff-aadb-23d19830ccdb)

![image](https://github.com/user-attachments/assets/11fdb83b-e4df-43c7-bbcb-e179ec6212de)

![image](https://github.com/user-attachments/assets/82cbb32f-2534-4eae-b1ce-e2c4c4f5a2c0)

![image](https://github.com/user-attachments/assets/21095e78-755a-40ce-b977-4979963aaa14)


