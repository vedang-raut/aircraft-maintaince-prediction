
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Sample dataset to train the model
data = {
    'Flight_Hours': [1000, 400, 1200, 200, 850, 1600, 300, 900, 1800, 100],
    'Landings': [500, 150, 600, 80, 400, 900, 100, 450, 950, 30],
    'Engine_Temp': [620, 540, 650, 500, 610, 680, 520, 630, 700, 490],
    'Vibration': [3.0, 1.2, 3.4, 1.0, 2.5, 3.8, 1.1, 3.1, 4.0, 0.9],
    'Last_Maintenance': [150, 60, 200, 40, 120, 250, 50, 130, 270, 30],
    'Maintenance_Needed': [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Prepare training data
X = df.drop('Maintenance_Needed', axis=1)
y = df['Maintenance_Needed']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("✈️ Aircraft Maintenance Predictor")
st.markdown("Enter flight data to predict if maintenance is needed.")

# User input
flight_hours = st.slider("Flight Hours", 0, 2000, 500)
landings = st.slider("Landings", 0, 1000, 250)
engine_temp = st.slider("Engine Temperature (°C)", 400, 800, 600)
vibration = st.slider("Vibration Level", 0.0, 5.0, 2.0)
last_maintenance = st.slider("Hours Since Last Maintenance", 0, 300, 100)

# Prediction
input_data = pd.DataFrame([[flight_hours, landings, engine_temp, vibration, last_maintenance]],
                          columns=['Flight_Hours', 'Landings', 'Engine_Temp', 'Vibration', 'Last_Maintenance'])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display result
if prediction == 1:
    st.error(f"⚠️ Maintenance **is likely needed**. Confidence: {probability:.2%}")
else:
    st.success(f"✅ No immediate maintenance needed. Confidence: {1 - probability:.2%}")

# Display input summary
st.markdown("### 🔍 Input Summary")
st.dataframe(input_data)
