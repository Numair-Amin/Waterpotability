#!/usr/bin/env python
# coding: utf-8

# ### Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Getting Water potability file
df_water = pd.read_csv('water_potability.csv')
df_water.head()

# Data Cleaning
print('NaN values in Water dataset:')
df_water.isna().sum()
df_water.info()

df_clean = df_water.fillna(df_water.mean())

print('Sum of Null values after cleaning = ', df_clean.isnull().sum().sum())
df_clean.info()

# ### Feature Selection
plt.figure(figsize=(10, 5))
sns.heatmap(df_clean.corr(), annot=True)

x = df_clean.drop("Potability", axis=1)
y = df_clean["Potability"]

# ### Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('No of records in training data (before SMOTE):', x_train.shape[0])
print('No of records in testing data:', x_test.shape[0])

# =========================
# Apply SMOTE on training set
# =========================
smote = SMOTE(random_state=42)
x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

print("Training data shape after SMOTE:", x_train_sm.shape)
print("Class distribution after SMOTE:", np.bincount(y_train_sm))

# ### Model Building with Random Forest
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(random_state=42)
model2.fit(x_train_sm, y_train_sm)

pred2 = model2.predict(x_test)

print(f'Accuracy: {round(100 * accuracy_score(y_test, pred2), 2)} %')

plt.figure(figsize=(5, 3))
sns.heatmap(confusion_matrix(y_test, pred2), annot=True, fmt='2g')

# Example prediction
sample = x_test.loc[1578, :].to_frame().T
pred = model2.predict(sample)
print("Prediction for sample row:", pred)

# Save trained model
joblib.dump(model2, "water_potability_model.pkl")
