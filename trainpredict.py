import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Load dataset
df = pd.read_csv('updated_data.csv')

# Convert datetime to index
df.set_index("datetime", inplace=True)

# Define features and target
features = ["Sales", "Passersby", "Transactions", "Conversion", "Capture Rate", "Traffic"]
target = "Traffic"

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create sequences for LSTM
def create_sequences(data, feature_count, target_index, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :feature_count])
        y.append(data[i + sequence_length, target_index])
    return np.array(X), np.array(y)

# Define feature and target indexes
feature_count = len(features)
target_index = df.columns.get_loc(target)

# Create sequences
sequence_length = 300
X, y = create_sequences(scaled_data, feature_count, target_index, sequence_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, feature_count), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer="adam", loss="mse")
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test), verbose=2)

# Predict
y_pred = model.predict(X_test)

# Reverse scaling for evaluation
def inverse_transform_column(data, target_index, scaler, feature_count):
    zeros_for_scaling = np.zeros((len(data), scaled_data.shape[1]))  # Match original data shape
    zeros_for_scaling[:, target_index] = data
    return scaler.inverse_transform(zeros_for_scaling)[:, target_index]

y_test_rescaled = inverse_transform_column(y_test, target_index, scaler, feature_count)
y_pred_rescaled = inverse_transform_column(y_pred.flatten(), target_index, scaler, feature_count)

# Evaluate
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print("Mean Squared Error:", mse)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Traffic", color="blue")
plt.plot(y_pred_rescaled, label="Predicted Traffic", color="orange")
plt.legend()
plt.title("Traffic Prediction with LSTM")
plt.xlabel("Time Steps")
plt.ylabel("Traffic")
plt.show()

# Function to generate future predictions
def predict_future(model, data, scaler, sequence_length, feature_count, target_index, steps_ahead):
    future_predictions = []
    current_sequence = data[-sequence_length:].copy()  # Use the last sequence from the dataset

    for _ in range(steps_ahead):
        next_value = model.predict(current_sequence[np.newaxis, :, :], verbose=0)
        future_predictions.append(next_value[0, 0])
        next_row = np.zeros(feature_count)
        next_row[target_index] = next_value
        current_sequence = np.append(current_sequence[1:], [next_row], axis=0)

    future_predictions_rescaled = inverse_transform_column(future_predictions, target_index, scaler, feature_count)
    return future_predictions_rescaled

# Predict future values
steps_ahead = 24
future_traffic = predict_future(model, scaled_data, scaler, sequence_length, feature_count, target_index, steps_ahead)

# Generate future dates
last_date = pd.to_datetime(df.index[-1])
future_dates = [last_date + pd.Timedelta(hours=i) for i in range(1, steps_ahead + 1)]

# Save future predictions to a CSV file
future_results_df = pd.DataFrame({
    "Datetime": future_dates,
    "Predicted Traffic": future_traffic
})
future_results_df.to_csv("future_predictions.csv", index=False)

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Traffic", color="blue")
plt.plot(y_pred_rescaled, label="Predicted Traffic", color="orange")
plt.plot(future_dates, future_traffic, label="Future Predictions", color="green")
plt.legend()
plt.title("Traffic Prediction and Future Forecasting")
plt.xlabel("Time")
plt.ylabel("Traffic")
plt.show()



model.save('train_predict.keras')
print("Model saved as 'traffic_model_daily_sequence.keras'.")
# Save metadata
metadata = {
    "features": features,
    "target": target,
    "sequence_length": sequence_length,
    "scaler": scaler,
    "last_sequence": scaled_data[-sequence_length:],  # Save the last sequence from the training data
    "last_date": df.index[-1]  # Save the last date from the dataset
}
with open("trainpredict.pkl", "wb") as f:
    pickle.dump(metadata, f)
print("Model and metadata saved.")