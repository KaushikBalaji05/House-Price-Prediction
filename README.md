# House Price Prediction using Linear Regression

## Overview

This project uses a Linear Regression model to predict house prices based on various housing features such as area, number of bedrooms, bathrooms, parking spaces, furnishing status, and other amenities.

The goal of this project was to understand the complete machine learning workflow, including data preprocessing, feature encoding, model training, evaluation, and interpretation of results.

---

## Dataset

The project uses the **Housing Dataset**, which contains information about 545 houses and their corresponding prices.

### Features

* Area
* Bedrooms
* Bathrooms
* Stories
* Main Road Access
* Guest Room
* Basement
* Hot Water Heating
* Air Conditioning
* Parking
* Preferred Area
* Furnishing Status

### Target Variable

* Price

---

## Project Workflow

### 1. Data Exploration

The dataset was inspected using Pandas to understand:

* Dataset dimensions
* Data types
* Statistical summaries
* Missing values

### 2. Data Preprocessing

The following preprocessing steps were performed:

* Converted binary categorical values (`yes`/`no`) into numerical values (`1`/`0`)
* Applied One-Hot Encoding to the `furnishingstatus` column
* Separated features and target variable

### 3. Train-Test Split

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

### 4. Model Training

A Linear Regression model from Scikit-Learn was trained on the processed dataset.

### 5. Evaluation

Model performance was evaluated using:

* Mean Absolute Error (MAE)
* R² Score

### 6. Visualization

A scatter plot was created to compare actual house prices against predicted house prices.

---

## Results

| Metric                    | Value   |
| ------------------------- | ------- |
| Mean Absolute Error (MAE) | 970,043 |
| R² Score                  | 0.653   |

The model is able to explain approximately 65% of the variation in house prices on the test dataset.

---

## Feature Insights

Based on the learned coefficients, some of the features with the strongest positive impact on house prices were:

* Number of Bathrooms
* Air Conditioning
* Hot Water Heating
* Preferred Area
* Number of Stories

Features such as unfurnished houses had a negative impact on the predicted price.

---

## Files

```text
Housing.csv                  Dataset
house_price.py               Main project file
house_price_model.pkl        Saved trained model
actual_vs_predicted.png      Prediction visualization
requirements.txt             Project dependencies
```

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib

---

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python house_price.py
```

---

## What I Learned

Through this project, I gained practical experience with:

* Data preprocessing
* Categorical feature encoding
* Train-test splitting
* Linear Regression
* Model evaluation
* Data visualization
* Interpreting model coefficients

This project served as my first end-to-end machine learning project and helped me understand how different stages of the ML pipeline fit together.
