import numpy as np
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



from DataLoad import df, X, y



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create Keras model
def create_model():
    model = Sequential([
        Input(shape=(X.shape[1],)),  # Input layer specifying input shape
        Dense(64, activation='relu'),  # Hidden layer with 64 neurons
        Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(64, activation='relu'),  # Hidden layer with 64 neurons
        Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
cv_scores = cross_val_score(create_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))
