from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv("cardio_train.csv")
X = df.drop(columns='cardio', axis=1)
y = df['cardio']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        glucose = int(request.form['glucose'])
        smoking = int(request.form['smoking'])
        alcohol = int(request.form['alcohol'])
        activity = int(request.form['activity'])
        cardio = int(request.form['cardio'])
        
        # Create the input data array
        input_data = np.array([[age,gender, height, weight,ap_hi,ap_lo,  cholesterol, glucose, smoking, alcohol, activity, cardio]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make a prediction
        prediction = model.predict(input_data_scaled)
        
        if prediction[0] == 0:
            result = 'The Person does not have a Heart Disease'
        else:
            result = 'The Person has Heart Disease'
        
        return render_template('index.html', prediction=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)