import pandas as pd
import numpy as np
import tensorflow as tf
from tfkan.layers import DenseKAN

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# Load data
file_path = '../data.csv'
data = pd.read_csv(file_path)

# Prepare features and target
X = data.drop(columns=['filename', 'result'])
y = data['result']

# Scale the features only
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define the model
model = tf.keras.models.Sequential([
    DenseKAN(256),
    DenseKAN(256),
    DenseKAN(256),
    DenseKAN(1)
])

model.build(input_shape=(None, X_train.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Add EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
)

# Train the model
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32,
    validation_split=0.2, callbacks=[early_stopping, reduce_lr]
)

# Predict on test data
y_test_predict = model.predict(X_test).flatten()
y_test_actual = y_test.values.flatten()

# Evaluate test performance
mae = mean_absolute_error(y_test_actual, y_test_predict)
mse = mean_squared_error(y_test_actual, y_test_predict)
rmse = mean_squared_error(y_test_actual, y_test_predict, squared=False)
r2 = r2_score(y_test_actual, y_test_predict)

print(f'Test Set Metrics:')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

# Export predictions to CSV files
# For the test set
test_results = pd.DataFrame({
    'y_test': y_test_actual,
    'y_test_predict': y_test_predict
})
test_results.to_csv('test_predictions.csv', index=False)
print('\nTest predictions saved to test_predictions.csv')
