import pandas as pd
import numpy as np
import tensorflow as tf
from tfkan.layers import DenseKAN

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import shap

# Upgrade/install SHAP if needed
# !pip install --upgrade shap

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
file_path = '../data.csv'
data = pd.read_csv(file_path)

# Drop non-feature columns and separate target
X = data.drop(columns=['filename', 'result'])
y = data['result']

# Scale the features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------
# 2. Build the Model
# -------------------
model = tf.keras.models.Sequential([
    DenseKAN(128),
    DenseKAN(128),
    DenseKAN(128),
    DenseKAN(1)
])

model.build(input_shape=(None, X_train.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
)

# ---------------
# 3. Train Model
# ---------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ----------------------------------------
# 4. Permutation-Based SHAP Explanations
# ----------------------------------------
# Using shap.explainers.Permutation for black-box models.
# This approach does feature permutations internally to estimate SHAP values.

# We create an explainer using the training set as reference.
explainer = shap.explainers.Permutation(model.predict, X_train)

# Now compute the SHAP values for the test set
shap_values = explainer(X_test)

# If you have feature names, it's useful to attach them:
feature_names = X.columns.tolist()
shap_values.feature_names = feature_names

# 4a. Beeswarm plot (shows per-feature distributions of SHAP values)
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, show=False)  # show=False so we can customize the figure if we want
plt.title("SHAP Beeswarm (Permutation-Based)")
plt.show()

# 4b. Bar plot of global feature importance
# (This aggregates the absolute SHAP values across all samples)
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
plt.title("Global Feature Importance (Permutation-Based SHAP)")
plt.show()
