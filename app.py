from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('house_model.joblib')

# Mappings
building_mapping={
"Apartment": 0,
"Bungalows": 1,
"Studio Apartment": 2,
"Villa": 3,
}

region_mapping={
"Andheri": 0,
"Bandra": 1,
"Dadar":2,
"Ghatkopar": 3,
"Lower Parel": 4,
"Mumbai": 5,
}

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # if request.method == 'POST':
        input_building_mapping=request.form['Types']
        input_region_mapping=request.form['Regions']
        input_bhk=int(request.form['BHK'])
        input_area=int(request.form['Area'])
        
        
        input_building_encoded = building_mapping.get(input_building_mapping, 0)
        input_region_encoded = region_mapping.get(input_region_mapping, 0)
        input_data = np.array([input_bhk,
        input_building_encoded,input_area,  input_region_encoded]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)
        
        return render_template('index.html', prediction=prediction)

    # return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
