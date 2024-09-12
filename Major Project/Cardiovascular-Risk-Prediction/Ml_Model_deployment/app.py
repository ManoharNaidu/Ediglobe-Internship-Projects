from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_ten_year_chd():
    age = request.form.get("age")
    total_cholesterol= float(request.form.get("total_cholesterol"))
    systolic_blood_pressure= float(request.form.get("systolic_blood_pressure"))
    diastolic_blood_pressure= float(request.form.get("diastolic_blood_pressure"))
    bmi= float(request.form.get("bmi"))
    heart_rate= float(request.form.get("heart_rate"))
    blood_glucose= float(request.form.get("blood_glucose"))

    #prediction
    input_1 = (age,total_cholesterol,systolic_blood_pressure,diastolic_blood_pressure,bmi,heart_rate,blood_glucose)
    input_array = np.asarray(input_1)
    input_reshape = input_array.reshape(1,-1)
    prediction_result = model.predict(input_reshape)
    if prediction_result[0]==0:
        result = "Not a heart disease"
    else:
        result = "A heart patient"
        
    return result
    
if __name__=='__main__':
    app.run(debug=True)