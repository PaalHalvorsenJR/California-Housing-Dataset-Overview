import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Last ned datasettet
print("Laster datasettet...")
california_data = fetch_california_housing(as_frame=True)
df = california_data.frame


print(df.head())  # Preview the dataset
print(df.columns)  # Print the column names

# Sjekk datasettet
print("Datasettet:")
print(df.head())

# 2. Forhåndsbehandle data
# Håndter manglende verdier (ingen i dette datasettet, men inkluderer steg for robusthet)
df.fillna(df.mean(), inplace=True)

# Funksjonsutvidelse (Feature Engineering)
df['Rooms_Per_Household'] = df['AveRooms'] / df['AveOccup']

# Opprett X (uavhengige variabler) og y (avhengig variabel)
X = df.drop(columns=['MedHouseVal'])  # MedHouseVal er boligprisene
y = df['MedHouseVal']

# Splitt dataen i trenings- og testsett
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaler data (standardisering)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Tren en XGBoost-modell
print("\nTrener XGBoost-modellen...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# 4. Evaluer modellen
y_pred = xgb_model.predict(X_test)

# Beregn R² og MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModellevaluering:")
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# 5. Visualiser resultater
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Faktiske priser")
plt.ylabel("Predikerte priser")
plt.title("Faktiske vs. Predikerte boligpriser")
plt.grid(True)
plt.show()
