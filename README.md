# PizzaTimeSeriesForecasting

## Project Overview
This project focuses on forecasting pizza sales using time series analysis. It aims to provide insights into sales dynamics, which can be used for strategic business planning, inventory management, and staffing.

- **Dataset**: The dataset was taken from kaggle. and contains 1 year worth of data about pizza orders. That translates into 48620 rows ordered by date and time.
- **Goal**: Predict sales of Hawaiian pizza for the next 30 days
- **Authors**: Maïca Muñoz Cuevas, Ignacio Ferro, Mohammed Alotaibi, Jose Carranque, Maria Jose Perez Gomez, Rodrigo Reyes

## Machine Learning Model

- **Random Forest Regressor**: Chosen for its strength in handling the complexity of non-linear relationships and interactions between features without extensive hyperparameter tuning.

- **RandomizedSearchCV**: Used for the cross-validation

## High-Level Data Preprocessing:
Our project begins with a robust preprocessing pipeline that transforms raw sales data into a clean, structured format ready for analysis. 
The preprocessing steps involve:
- **Data Cleaning**: We address missing values, outliers, and duplicates to ensure the integrity of our dataset.
- **Feature Extraction**: From the timestamp data, we extract meaningful features such as day of the week, time of day, and whether the date is a holiday, as these factors significantly influence sales patterns.
- **Normalization/Scaling**: Numerical features are scaled to a uniform range to enhance the performance of our machine learning algorithms, particularly for distance-based models.

## Project Structure
This repository is organized to facilitate easy navigation and understanding, ensuring that stakeholders can interact with and explore the project with ease:

- `/data`: Contains the `pizza_orders.csv` dataset used for analysis and modeling. This file encompasses a year's worth of pizza order data, offering a comprehensive view into sales patterns and customer preferences.
- `/models`: Houses saved trained machine learning models, including our primary Random Forest and Linear Regression models, enabling easy access and deployment for future forecasting.
- `/notebooks`: Features the `Pizza_Sales_Analysis.ipynb` Jupyter notebook, which details our exploratory data analysis, feature engineering, and the iterative process of model training and evaluation.
- `/scripts`: Python scripts for automating the model training, prediction, and plotting processes, streamlining the workflow from raw data to actionable insights.
- `/app`: Contains Streamlit application files for an interactive visualization and prediction interface, enhancing the stakeholder experience by making our findings accessible and actionable.

## Installation and Setup
Setting up this project on your local machine is straightforward. Begin by cloning the repository and installing the required dependencies:

```bash
# Clone the project repository
git clone https://github.com/your-username/PizzaTimeSeriesForecasting.git

# Navigate to the project directory
cd PizzaTimeSeriesForecasting

# Install the necessary dependencies
pip install -r requirements.txt

