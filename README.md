# Car Price Prediction using Machine Learning

## Project Overview

This project focuses on predicting the selling prices of used cars using Machine Learning techniques. It was developed as part of **Task 3 of the CodeAlpha Data Science Internship**.

The project uses various car-related attributes, including manufacturing year, present price, kilometres driven, fuel type, selling type, transmission, and ownership details, to estimate the selling price of a vehicle.

Two regression algorithms, Linear Regression and Random Forest Regression, were implemented and evaluated to identify the better-performing model.

## Objectives

- Understand and explore the used car dataset.
- Perform data cleaning and preprocessing.
- Handle duplicate records.
- Conduct Exploratory Data Analysis (EDA).
- Perform feature engineering.
- Build regression models for price prediction.
- Evaluate and compare model performance.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Dataset Description

The project uses a used car dataset containing **301 records and 9 columns**.

| Column | Description |
|---|---|
| Car_Name | Name of the car |
| Year | Manufacturing year |
| Selling_Price | Selling price of the car |
| Present_Price | Current/ex-showroom price feature in the dataset |
| Driven_kms | Kilometres driven |
| Fuel_Type | Fuel type |
| Selling_type | Dealer or individual selling type |
| Transmission | Manual or automatic transmission |
| Owner | Number of previous owners |

The dataset was checked for missing values and duplicate records. Two duplicate records were removed, resulting in 299 records for analysis.

## Project Workflow

### 1. Data Loading
Loaded the dataset using Pandas and examined its structure, dimensions, column names, and data types.

### 2. Data Cleaning and Preprocessing
- Checked for missing values.
- Identified and removed duplicate records.
- Prepared numerical and categorical features.
- Used median imputation for numerical features.
- Used most-frequent imputation and one-hot encoding for categorical features.

### 3. Exploratory Data Analysis
Performed data exploration and created visualizations to understand:
- Selling price distribution.
- Relationship between present price and selling price.
- Average selling prices across fuel types.
- Actual versus predicted selling prices.

### 4. Feature Engineering
Created a new feature named `Vehicle_Age` using the difference between the maximum manufacturing year in the dataset and each car's manufacturing year.

### 5. Model Building
Implemented two regression algorithms:

- Linear Regression
- Random Forest Regressor

The dataset was divided into training and testing sets using an 80:20 split.

### 6. Model Evaluation
Evaluated the models using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Model Performance

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 1.4121 | 1.3688 |
| RMSE | 2.4316 | 3.2628 |
| R² Score | 0.7706 | 0.5869 |

**Selected Model: Linear Regression**

Linear Regression was selected because it achieved a higher R² score and lower RMSE on the test dataset. Random Forest achieved a slightly lower MAE, so the comparison considered all three evaluation metrics rather than relying on a single metric.

## Project Structure

```text
CodeAlpha_Car_Price_Prediction/
│
├── Dataset/
│   ├── car_data_original.csv
│   └── car_data_cleaned.csv
│
├── Notebooks/
│   └── CodeAlpha_Task3_Car_Price_Prediction.ipynb
│
├── src/
│   ├── 01_data_preprocessing.py
│   └── 02_car_price_prediction.py
│
├── Reports/
│   └── CodeAlpha_Task3_Car_Price_Prediction_Report.pdf
│
├── Visualizations/
│   ├── selling_price_distribution.png
│   ├── selling_vs_present_price.png
│   ├── average_price_by_fuel.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── model_outputs/
│   ├── model_evaluation_metrics.csv
│   ├── test_predictions.csv
│   ├── permutation_feature_importance.csv
│   └── model_metrics.txt
│
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Folder

```bash
cd CodeAlpha_Car_Price_Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Preprocessing Script

```bash
python src/01_data_preprocessing.py
```

### 5. Run the Machine Learning Script

```bash
python src/02_car_price_prediction.py
```

Alternatively, open the Jupyter Notebook in Jupyter Notebook or JupyterLab and execute the cells sequentially.

## Key Learning Outcomes

- Practical understanding of regression algorithms.
- Experience with data cleaning and preprocessing.
- Understanding categorical encoding and numerical imputation.
- Application of feature engineering.
- Comparing Machine Learning models using evaluation metrics.
- Creating meaningful data visualizations.

## Limitations

- The dataset contains a relatively small number of records.
- Some potentially useful factors, such as vehicle condition, service history, and location, are not included.
- Predictions are based on the available dataset and should not be treated as guaranteed real-world market valuations.

## Internship Information

**Internship Organization:** CodeAlpha  
**Domain:** Data Science  
**Task:** Task 3 – Car Price Prediction using Machine Learning

## Acknowledgement

I would like to thank **CodeAlpha** for providing this internship opportunity and helping me gain practical experience in Data Science and Machine Learning.

---

**Developed as part of the CodeAlpha Data Science Internship.**
