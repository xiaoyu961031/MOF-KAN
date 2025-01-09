import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Configure plot parameters
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# Load and preprocess data
file_path = '../data.csv'
data = pd.read_csv(file_path)

# Prepare features and target
X = data.drop(columns=['filename', 'result'])
y = data['result']

# Scale the features only
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = y.values.reshape(-1, 1)
# Split data into training and testing sets
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Define parameter grids for each model
param_grids = {
    'MLP Regressor': {'hidden_layer_sizes': [(128,128,128)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}
}

# Define learners
learners = {
    'MLP Regressor': MLPRegressor(random_state=42, max_iter=500)
}

# Evaluate each learner with hyperparameter tuning
for name, model in learners.items():
    print(f'Evaluating {name}...')
    grid_search = GridSearchCV(
        model, param_grids[name], scoring='neg_mean_absolute_error', cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train_scaled.ravel())
    
    # Best model and predictions
    best_model = grid_search.best_estimator_
    y_test_pred_scaled = best_model.predict(X_test).reshape(-1, 1)
    y_train_pred_scaled = best_model.predict(X_train).reshape(-1, 1)
    
    # Evaluation metrics on test set
    mae = mean_absolute_error(y_test_scaled, y_test_pred_scaled)
    mse = mean_squared_error(y_test_scaled, y_test_pred_scaled)
    
    # Print test set results
    print(f'{name} Evaluation:')
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Mean Absolute Error (MAE) on test set: {mae}')
    print(f'Mean Squared Error (MSE) on test set: {mse}')

    test_results = pd.DataFrame({
        'y_test': y_test_scaled.flatten(),
        'y_test_predict': y_test_pred_scaled.flatten()
    })
    
    model_name_sanitized = name.replace(' ', '_')
    test_csv_filename = f'{model_name_sanitized}_test_predictions.csv'
    test_results.to_csv(test_csv_filename, index=False)
    
    print(f'Test predictions saved to {test_csv_filename}')
    print('=' * 50)
