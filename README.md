# PizzaTimeSeriesForecasting

## Project Overview
This project focuses on forecasting pizza sales using time series analysis. It aims to provide insights into sales dynamics, which can be used for strategic business planning, inventory management, and staffing.

- **Dataset**: The dataset was taken from kaggle. and contains 1 year worth of data about pizza orders. That translates into 48620 rows ordered by date and time.
- **Goal**: The goal is to accurately predict weekly pizza sales, enabling effective staff scheduling and inventory management
- **Authors**: Maïca Muñoz Cuevas, Ignacio Ferro, Mohammed Alotaibi, Jose Carranque, Maria Jose Perez Gomez, Rodrigo Reyes

## Machine Learning Model


- **Random Forest Regressor**: Chosen for its strength in handling the complexity of non-linear relationships and interactions between features without extensive hyperparameter tuning.

- **Linear Regression**: Used as a baseline to benchmark the performance of more complex models.

- **Time Series Forecasting**: Special attention is given to time series forecasting models to predict future sales based on historical patterns, taking into account seasonal trends and cyclical behavior inherent in the sales data.

## High-Level Data Preprocessing:
Our project begins with a robust preprocessing pipeline that transforms raw sales data into a clean, structured format ready for analysis. 
The preprocessing steps involve:
- **Data Cleaning**: We address missing values, outliers, and duplicates to ensure the integrity of our dataset.
- **Feature Extraction**: From the timestamp data, we extract meaningful features such as day of the week, time of day, and whether the date is a holiday, as these factors significantly influence sales patterns.
- **Normalization/Scaling**: Numerical features are scaled to a uniform range to enhance the performance of our machine learning algorithms, particularly for distance-based models.

## Project Structure
This repository is organized to facilitate easy navigation and understanding:

- `/data`: Contains the dataset used for analysis and modeling.
- `/models`: Saved trained machine learning models.
- `/notebooks`: Jupyter notebooks for exploratory data analysis and model development.
- `/scripts`: Python scripts for automating the model training, prediction, and plotting processes.
- `/app`: Streamlit application files for interactive visualization and prediction.

## Installation and Setup
To set up this project on your local machine, follow these steps:

```bash
# Clone the project repository
git clone https://github.com/your-username/PizzaTimeSeriesForecasting.git

# Navigate to the project folder
cd PizzaTimeSeriesForecasting

# Install the necessary dependencies
pip install -r requirements.txt

