# Pre-Owned Vehicle Market Price Predictor 🚗📊

A machine learning pipeline and interactive web application designed to estimate the fair market value of pre-owned vehicles. Leveraging historical listing data, this tool provides instant, data-driven price valuations to assist in evaluating trade-in rates, acquisition costs, and resale potential within the automotive market.

![Streamlit App Interface](Screenshot%20(62).png) 

## Project Overview
This repository contains the end-to-end development of a predictive pricing model, from initial exploratory data analysis (EDA) and feature engineering to model deployment. The final deliverable is a functional Streamlit web interface that allows users to input specific vehicle specifications and receive an immediate price estimation.

## Features
* **Exploratory Data Analysis (EDA):** Comprehensive univariate and bivariate analysis, skewness correction, and correlation mapping to identify primary price drivers.
* **Feature Engineering & Selection:** Categorical encoding, standardization, and K-best feature selection to optimize the dataset for machine learning.
* **Machine Learning Integration:** Model selection, training, and hyperparameter tuning utilizing a Gradient Boosting regressor.
* **Interactive UI:** A deployed Streamlit application bridging the backend Python model with a user-friendly frontend interface.

## Tech Stack
* **Language:** Python 3
* **Data Processing & Analysis:** pandas, numpy, scipy
* **Machine Learning:** scikit-learn
* **Data Visualization:** matplotlib, seaborn
* **Model Serialization:** joblib
* **Web Deployment:** Streamlit

## Repository Structure
* `Major_Project_BaseNoteBook.ipynb`: The core Jupyter Notebook containing all data cleaning, EDA, model training, and evaluation logic.
* `app.py`: The Streamlit web application script.
* `best_model.pkl`: The serialized, pre-trained machine learning model.
* `brand_encoder.pkl` / `final_features.pkl`: Saved preprocessing objects required for the web app's input pipeline.
* `car_price_dataset.csv`: The historical vehicle listing dataset.
