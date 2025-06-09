# 🚗 Task 03: Car Price Prediction with Machine Learning

This project is part of my Data Science Internship at **Oasis Infobyte** under the **AICTE OIB-SIP June 2025** cohort.

---

## 📌 Problem Statement

Car prices are influenced by several factors such as fuel type, transmission, brand, mileage, and more. In this project, I built and compared multiple regression models to predict the **selling price** of used cars based on their features.

---

## 🗂️ Dataset Used

- `car data.csv`: Contains details about used cars including:
  - Car Name, Year, Present Price, Selling Price, Driven kms
  - Fuel Type, Seller Type, Transmission, Owner

---

## 🛠️ Technologies & Libraries

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn (Linear Regression, Random Forest, Gradient Boosting)
- XGBoost
- GridSearchCV

---

## ✅ Project Highlights

- **Preprocessing**: 
  - Categorical encoding with `pd.get_dummies`
  - Train-test split with `train_test_split`

- **Modeling**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBRegressor (Extreme Gradient Boosting)
  - Hyperparameter tuning with GridSearchCV

- **Evaluation**:
  - R² Score
  - Mean Squared Error (MSE)
  - Feature Importance Comparison

- **Visualizations**:
  - Price Distribution Histogram
  - Scatter Plot (Driven kms vs Price)
  - Feature Importances from each model
  - Actual vs Predicted Prices
  - Average Price by Fuel Type

All plots are saved in the `figures/` directory.

---

## 📊 Model Performance Comparison

| Model               | R² Score | Mean Squared Error |
|--------------------|----------|---------------------|
| Linear Regression  | 0.60     | 9.22                |
| Random Forest      | 0.97     | 0.71                |
| Gradient Boosting  | 0.96     | 0.88                |
| XGBRegressor       | 0.96     | 0.93                |

✅ **Best Model**: Random Forest Regressor with R² ≈ 0.97

---

## 📁 Project Structure

```bash
Task-03-Car-Price-Prediction/
├── car_price_prediction.py
├── Data/
│ └── car data.csv
├── figures/
│ ├── feature_importance.png
│ ├── price_disturbution.png
│ ├── scatter_plot.png
│ ├── feature_importance_rf.png
│ ├── actual_vs_predicted.png
│ ├── avg_price_fuel.png
│ ├── feature_importance_xgb.png
└── README.md
```

---

## 📽️ Video Demonstration

🔗 *(Coming soon)* — I will add a LinkedIn demo video link once published.

---

## 📌 Internship Info

- Program: AICTE OIB-SIP June 2025  
- Organization: Oasis Infobyte  
- Domain: Data Science  
- Task: 03 of 05

---

## 🔗 Connect

- 🔗 [LinkedIn Profile](https://github.com/alireza-taheriF)
- 💻 [GitHub](https://github.com/alireza-taheriF/OIBSIP)
- 🔖 #oasisinfobyte #datascience #machinelearning #python #carpriceprediction #internship #regressionmodels
