from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the datasets
crimes_df = pd.read_csv('data/Crimes_Dataset.csv')
suspects_df = pd.read_csv('data/Suspects_Dataset.csv')

# Preprocessing the Crimes dataset
crimes_df = crimes_df.dropna(subset=['Region', 'Time of Day', 'Evidence Found', 'Crime Weapon'])
crimes_df['Region'] = crimes_df['Region'].str.lower()
crimes_df['Time of Day'] = crimes_df['Time of Day'].str.lower()
crimes_df['Evidence Found'] = crimes_df['Evidence Found'].str.lower()
crimes_df['Crime Weapon'] = crimes_df['Crime Weapon'].str.lower()

# Filter only relevant columns for classification
filtered_crimes = crimes_df[(crimes_df['Region'] == 'village') &
                            (crimes_df['Time of Day'] == 'day') &
                            (crimes_df['Evidence Found'] == 'bones') &
                            (crimes_df['Crime Weapon'] == 'knife')]

# Encode categorical variables for classifier
le_monster = LabelEncoder()
crimes_df['Monster involved'] = le_monster.fit_transform(crimes_df['Monster involved'].astype(str))

# Preparing features and target for classification
X = crimes_df[['Region', 'Time of Day', 'Evidence Found', 'Crime Weapon']]
y = crimes_df['Monster involved']

# Convert categorical features into numeric
X = pd.get_dummies(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Neural Network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(le_monster.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
eval_results = model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", eval_results[1])

# Predicting the monster involved in filtered crimes
filtered_crimes_encoded = pd.get_dummies(filtered_crimes[['Region', 'Time of Day', 'Evidence Found', 'Crime Weapon']])
filtered_crimes_encoded = filtered_crimes_encoded.reindex(columns=X.columns, fill_value=0)
filtered_crimes_encoded = filtered_crimes_encoded.astype('float32')

predicted_monsters_nn = model.predict(filtered_crimes_encoded)
predicted_monsters_labels = np.argmax(predicted_monsters_nn, axis=1)
filtered_crimes['Predicted Monster'] = le_monster.inverse_transform(predicted_monsters_labels)

# Display the predictions
print("Predicted Monsters for Filtered Crimes:")
print(filtered_crimes[['Date', 'Region', 'Crime Type', 'Predicted Monster']])
filtered_crimes[['Date', 'Region', 'Crime Type', 'Predicted Monster']].to_csv('predicted_monsters.csv', index=False)
print("The predictions have been saved to 'predicted_monsters.csv'.")


