# Electricity Price Prediction

This repository contains a comprehensive analysis of hourly electricity prices for a data center located in Ireland. The project focuses on utilizing time series analysis techniques to develop predictive models that accurately forecast electricity prices. By analyzing historical price data and incorporating various influencing factors such as temperature, wind energy production, national system load, and wind speed, the goal is to produce precise and trustworthy forecasts.

## Project Overview

- **Objective**: Predict hourly electricity prices for a data center in Ireland to aid in planning and decision-making.
- **Dataset**: The dataset includes hourly electricity prices along with independent variables like temperature, wind energy production, system load, and more.
- **Methodology**:
  - Data Cleaning: Ensure data integrity by handling missing values and inconsistencies.
  - Exploratory Data Analysis (EDA): Analyze dataset features, check for stationarity, and understand correlations.
  - Modeling Approach: Develop various models including base models, multiple linear regression, and ARIMA(SARIMA) models.
  - Model Evaluation: Assess model performance using metrics such as MSE, RMSE, MAE, and diagnostic tests.
- **Final Model Selection**: Choose the SARIMAX model based on its performance metrics, although noting its limitations.
- **Forecasting**: Develop a custom forecast function for the selected SARIMA model to make predictions.

## Repository Structure

- `electricity_prices.csv`: Contains the dataset used for analysis.
- `FPT_.py`: Python file detailing the data preprocessing, exploratory data analysis, modeling, and evaluation steps.
- `readme_.txt`: Here you will find instructions on how to run FTP_.py file.
