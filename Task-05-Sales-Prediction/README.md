# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
data_path = 'Data/Advertising.csv'
df = pd.read_csv(data_path)

# Display few first rows
print('First few rows of the dataset')
print(df.head())

# Check the basic info
print('/nDataset Info: ')
print(df.info())
print('/nMissing Values: ')
print(df.isnull().sum())

# Drop unnecessary column (Unnamed: 0)
df = df.drop('Unnamed: 0', axis=1)

# Assume columns: 'TV', 'Radio', 'Newspaper' (advertising spend), 'Sales' (target)
X = df[['TV', 'Radio', 'Newspaper']]    # Features
y = df['Sales'] # Target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('/nModel Evaluation: ')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Sales Distribution
feature_importance = pd.DataFrame({'Feature': ['TV', 'Radio', 'Newspaper'], 'Importance': abs(model.coef_)})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Sales Prediction')
plt.savefig('figures/feature_importance.png')
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Prediction Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.savefig('figures/actual_vs_predicted.png')
plt.show()

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('/nRandom Forest Model Evaluation: ')
print(f'Mean Squared Model Evaluation: ')
print(f'Mean Squared Error: {mse_rf}')
print(f'R² score: {r2_rf}')

# Feature Importance Plot for Random Forest
feature_importance_rf = pd.DataFrame({'Feature' : ['TV', 'Radio', 'Newspaper'], 'Importance': rf_model.feature_importances_})
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_rf)
plt.title('Feature Importance in Sales Prediction (Random Forest)')
plt.savefig('figures/feature_importance_rf.png')
plt.show()

# Optimization with GridSearchCV
param_grid = {
              'n_estimators': [100, 200],
              'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('/nBest Parameters: ', grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)
print('/nBest Random Forest Model Evaluation: ')
print(f'R² Score: {r2_score(y_test, y_pred_best)}')

# Residuals distribution
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='darkred')
plt.title('Residuals Distribution - Linear Regression')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.savefig('figures/residuals_linear.png')
plt.show()

# Create a combined feature importance DataFrame
feature_importance_all = pd.DataFrame({
    'Feature': ['TV', 'Radio', 'Newspaper'],
    'Linear Regression': abs(model.coef_),
    'Random Forest': rf_model.feature_importances_,
    'Optimized RF': best_rf_model.feature_importances_
})

# Melt to long format for seaborn
melted = pd.melt(feature_importance_all, id_vars='Feature', var_name='Model', value_name='Importance')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Model', data=melted)
plt.title('Feature Importance Comparison')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('figures/feature_importance_comparison.png')
plt.show()

# Sample test
sample = np.array([[150, 20, 10]])  # sample ad budget
sample_scaled = scaler.transform(sample)  # use the same scaler as training
pred = best_rf_model.predict(sample_scaled)

print(f'\n📌 Predicted Sales for TV=150, Radio=20, Newspaper=10 => {pred[0]:.2f} units')

```bash
.
├── Data/
│ └── Advertising.csv
├── figures/
│ ├── actual_vs_predicted.png
│ ├── feature_importance.png
│ ├── feature_importance_rf.png
│ ├── feature_importance_comparison.png
│ ├── residuals_linear.png
├── sales_prediction.py
└── README.md
```


---

## ▶️ Usage

```bash
# Run the project
python sales_prediction.py
```
All results will be saved in figures/ and printed in the terminal.

```pyhton
# Predict for new ad budget
sample = np.array([[150, 20, 10]])
sample_scaled = scaler.transform(sample)
pred = best_rf_model.predict(sample_scaled)
print(f"Predicted Sales: {pred[0]:.2f}")
```
💡 Result: ~18.53 units sold


