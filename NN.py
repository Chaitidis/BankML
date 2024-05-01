from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout




# from DataLoad import df, X, y

# csv_file_name = 'Bank_data.csv'
df = pd.read_csv('Bank_data.csv')
# print(df.head())  # Display the first few rows of the DataFrame
# num_columns = df.shape[1]
# print("Number of columns:", num_columns)


X = df.drop(columns=['y'])
y = df['y']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential([
    Input(shape=(X.shape[1],)),  # Input layer specifying input shape
    Dense(16, activation='relu'),  # Hidden layer with 64 neurons
    Dropout(0.2),  # Dropout layer to prevent overfitting
    Dense(16, activation='relu'),  # Hidden layer with 64 neurons
    Dropout(0.2),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])



# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=27, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
with open('test_accuracy_nn.txt', 'w') as f:
    f.write(str(accuracy))

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

