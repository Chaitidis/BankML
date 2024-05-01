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
from tensorflow.keras import regularizers



# from DataLoad import df, X, y

excel_file_name = 'Bank_data.xlsx'
df = pd.read_excel(excel_file_name)
print(df.head())  # Display the first few rows of the DataFrame
# num_columns = df.shape[1]
# print("Number of columns:", num_columns)


X = df.drop(columns=['y'])
y = df['y']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture with L2 regularization
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
with open('test_accuracy_nn_l2.txt', 'w') as f:
    f.write(str(accuracy))

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
