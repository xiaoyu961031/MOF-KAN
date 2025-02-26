import pandas as pd
import numpy as np
import tensorflow as tf
from tfkan.layers import DenseKAN
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
from tcn import TCN

from sklearn.model_selection import train_test_split

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

file_path = '../data.csv'
data = pd.read_csv(file_path)

# Prepare features and target
X = data.drop(columns=['filename', 'result'])
y = data['result']

# Scale the features only
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled = np.expand_dims(X_scaled, axis=1)  
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
        
class KAN_TCN(models.Model):
    def __init__(self, input_dim, filters, kernel_size, dilation_rate, layer_units):
        super(KAN_TCN, self).__init__()
        self.tcn = TCN(nb_filters=filters, kernel_size=kernel_size, dilations=[dilation_rate], padding='causal', return_sequences=False, use_skip_connections=True, kernel_initializer='he_normal')
        self.dense_kan1 = DenseKAN(512)
        self.dense_kan2 = DenseKAN(512)
        self.dense_kan3 = DenseKAN(512)
        self.dense_kan4 = DenseKAN(1) 

    def call(self, inputs):
        x = self.tcn(inputs)
        x = self.dense_kan1(x)
        x = self.dense_kan2(x)
        x = self.dense_kan3(x)
        x = self.dense_kan4(x)
        return x
        
input_dim = X_train.shape[2]
filters = 64
kernel_size = 3
dilation_rate = 2
layer_units = 512

kan_tcn = KAN_TCN(input_dim, filters, kernel_size, dilation_rate, layer_units)
kan_tcn.build(input_shape=(None, None, input_dim))
kan_tcn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
kan_tcn.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

history = kan_tcn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

y_test_predict = kan_tcn.predict(X_test).flatten()
y_test_actual = y_test.values.flatten()

# Evaluate test performance
mae = mean_absolute_error(y_test_actual, y_test_predict)
mse = mean_squared_error(y_test_actual, y_test_predict)

print(f'Test Set Metrics:')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')

# Export predictions to CSV files
# For the test set
test_results = pd.DataFrame({
    'y_test': y_test_actual,
    'y_test_predict': y_test_predict
})
test_results.to_csv('test_predictions.csv', index=False)
print('\nTest predictions saved to test_predictions.csv')

