<h1>Car-Dheko---Used-Car-Price-Prediction Scripts Overview</h1>
Here's a summary of the various scripts designed for processing, cleaning, analyzing, and modeling car price prediction data:

<h2>1. Car Extraction Script:</h2>

<h3>Description:</h3> 
Processes and cleans Excel dataset (For eg.,kolkata_cars.xlsx) containing malformed dictionary strings.

<h3>Steps:</h3>

<ul><li>Load data into a pandas DataFrame.</li>
<li>Clean malformed dictionary strings.</li>
<li>Flatten nested dictionaries.</li>
<li>Extract key-value pairs from 'top' field.</li>
<li>Save cleaned data to Kolkata.xlsx.</li></ul>

<h3>Requirements:</h3> Python 3.x, pandas, openpyxl.

<h3>Installation:</h3> pip install pandas openpyxl

<h3>Usage:</h3> Adjust file path and run the script.

<h2>2. Car Data Concatenation Script</h2>
<h3>Description:</h3> Consolidates car data from multiple city-specific Excel files into one dataset.
<h3>Steps:</h3>
<ul><li>Read data from city-specific files (Banglore.xlsx, Kolkata.xlsx, etc.).</li>
<li>Select relevant columns.</li>
<li>Concatenate data into a single DataFrame.</li>
<li>Save combined data to cars_dataset.xlsx.</li></ul>
<h3>Requirements:</h3> Python 3.x, pandas, openpyxl.
<h3>Installation:</h3> pip install pandas openpyxl
<h3>Usage:</h3> Ensure files are in the correct directory and run the script.
<h2>3. Car Data Cleaning Script</h2>
<h3>Description:</h3> Cleans the consolidated car dataset by standardizing key fields and handling missing values.
<h3>Steps:</h3><ul><li>
Load data from cars_dataset.xlsx.</li>
<li>Clean and convert price, mileage, kilometers driven, engine specifications.</li>
<li>Handle missing values.</li>
<li>Remove unnecessary columns.</li>
<li>Save cleaned data to cleaned_cars_dataset.xlsx.</li></ul>
<h3>Requirements:</h3> Python 3.x, pandas, openpyxl, re.
<h3>Installation:</h3> pip install pandas openpyxl
<h3>Usage:</h3> Ensure dataset file is in the correct directory and run the script.
<h2>4. Car Cleaning2 Preprocess</h2>
<h3>Description:</h3> Further preprocesses the cleaned dataset by handling missing values, encoding categorical variables, normalizing numerical columns, and removing outliers.
<h3>Steps:</h3><ul>
<li>Load cleaned_cars_dataset.xlsx.</li>
<li>Handle missing values and encode categorical variables.</li>
<li>Normalize numerical columns using MinMaxScaler.</li>
<li>Remove outliers using IQR method.</li>
<li>Save preprocessed data to preprocessed_2_cars_dataset.xlsx.</li></ul>
<h3>Requirements:</h3> Python 3.x, pandas, numpy, scikit-learn, scipy, joblib.
<h3>Installation:</h3> pip install pandas numpy scikit-learn scipy joblib
<h3>Usage:</h3> Update file paths and run the script.
<h2>5. Eda1</h2>
<h3>Description:</h3> Performs exploratory data analysis (EDA) on the dataset, including scatter plots, histograms, box plots, and a correlation heatmap.
<h3>Steps:</h3><ul>
<li>Load dataset from preprocessed_2_cars_dataset.xlsx.</li>
<li>Display descriptive statistics.</li>
<li>Create scatter plots, histograms, box plots.</li>
<li>Generate a correlation heatmap.</li></ul>
<h3>Requirements:</h3> pandas, seaborn, matplotlib, openpyxl.
<h3>Installation:</h3> pip install pandas seaborn matplotlib openpyxl
<h3>Usage:</h3> Update file path and run the script.
<h2>6. Eda2</h2>
<h3>Description:</h3> Performs detailed analysis on car prices, including descriptive statistics, correlation analysis, and feature importance using RandomForestRegressor.
<h3>Steps:</h3><ul>
<li>Load dataset from preprocessed_2_cars_dataset.xlsx.</li>
<li>Calculate descriptive statistics.</li>
<li>Perform correlation analysis.</li>
<li>Assess feature importance using RandomForestRegressor.</li></ul>
<h3>Requirements:</h3> pandas, sklearn, numpy.
<h3>Installation:</h3> pip install pandas scikit-learn numpy openpyxl
<h3>Usage:</h3> Update file path and run the script.
<h2>7. Model</h2>
<h3>Description:</h3> Evaluates multiple machine learning models for car price prediction, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and K-Nearest Neighbors Regressor.
<h3>Steps:</h3><ul>
<li>Load dataset from preprocessed_2_cars_dataset.xlsx.</li>
<li>Split data into training and testing sets.</li>
<li>Train and evaluate models with and without hyperparameter tuning.</li>
<li>Compare model performance using metrics like MAE, MSE, and R2.</li></ul>
<h3>Requirements:</h3> pandas, scikit-learn, openpyxl.
<h3>Installation:</h3> pip install pandas scikit-learn openpyxl
<h3>Usage:</h3> Update file path and run the script.
<h2>8. Gradient Boosting</h2>
<h3>Description:</h3> Builds and evaluates a Gradient Boosting Regressor model for car price prediction, including hyperparameter tuning and saving/loading the model using pickle.
<h3>Steps:</h3><ul>
<li>Load dataset from preprocessed_2_cars_dataset.xlsx.</li>
<li>Prepare data for training.</li>
<li>Perform grid search for hyperparameter tuning.</li>
<li>Train and evaluate the Gradient Boosting model.</li>
<li>Save/load the model using pickle.</li></ul>
<h3>Requirements:</h3> pandas, numpy, scikit-learn, openpyxl, pickle.
<h3>Installation:</h3> pip install pandas numpy scikit-learn openpyxl
<h3>Usage:</h3> Update file path and run the script.
<br><br>
<hr>
<b>These scripts collectively cover the entire workflow from data extraction and cleaning to model evaluation and prediction for car price analysis. Adjust paths and column names as needed based on your dataset.</b>
<br><br>
<ul><li>s_l.py: The main Streamlit application file.</li>
<li>id.xlsx: Excel file with variant names and IDs.</li>
<li>cleaned_cars_dataset.xlsx: Excel file with cleaned car data.</li>
<li>pic2.pkl: Pickle file containing the pre-trained model.</li>
<li>cars1.png: Background image used in the app.</li>
<li>image_to_base64(img_path): Converts an image file to a base64-encoded string for use in CSS.</li>
<li>fixthecode(val, max, min): Normalizes the input values for the model.</li>
<li>Rfixthecode(val, max, min): Rescales the predicted price back to the original range.</li>
<li>predict_price(model, input_data): Uses the pre-trained model to predict the car price based on user inputs.</li>
<li>Streamlit UI: Provides the user interface for input collection and displays the predicted price.</li></ul>


<hr>

![image](https://github.com/user-attachments/assets/bfddceac-8ed5-48df-b2dc-3aae6ecf2e3a)

![image](https://github.com/user-attachments/assets/8242e7c6-4b78-4336-895b-7005c6a1b106)

![image](https://github.com/user-attachments/assets/fdcaff15-d7b5-4b17-b194-4ff1987581d0)

![image](https://github.com/user-attachments/assets/9dfa73a3-da20-48ff-aadb-23d19830ccdb)

![image](https://github.com/user-attachments/assets/11fdb83b-e4df-43c7-bbcb-e179ec6212de)

![image](https://github.com/user-attachments/assets/82cbb32f-2534-4eae-b1ce-e2c4c4f5a2c0)

![image](https://github.com/user-attachments/assets/21095e78-755a-40ce-b977-4979963aaa14)


