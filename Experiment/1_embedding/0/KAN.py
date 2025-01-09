import pandas as pd
import numpy as np
import tensorflow as tf
from tfkan.layers import DenseKAN
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib

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

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class KAN_Transformer(models.Model):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim):
        super(KAN_Transformer, self).__init__()
        self.embedding = layers.Dense(embed_dim)
        self.transformer = TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense_kan1 = DenseKAN(512)
        self.dense_kan2 = DenseKAN(512)
        self.dense_kan3 = DenseKAN(512)
        self.dense_kan4 = DenseKAN(1)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer(x)
        x = self.global_pool(x)
        x = self.dense_kan1(x)
        x = self.dense_kan2(x)
        x = self.dense_kan3(x)
        x = self.dense_kan4(x)
        return x


input_dim = X_train.shape[2] 
embed_dim = 64
num_heads = 4
ff_dim = 128
layer_units = 128


kan_transformer = KAN_Transformer(input_dim, embed_dim, num_heads, ff_dim)
kan_transformer.build(input_shape=(None, None, input_dim))
kan_transformer.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


kan_transformer.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Fit the model with the callbacks
history = kan_transformer.fit(
    X_train, y_train, 
    epochs=200, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping, reduce_lr]
)


y_test_predict = kan_transformer.predict(X_test).flatten()
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

