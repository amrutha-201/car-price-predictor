from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('cleaned_car.csv')

@app.route('/')
def index():
    
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    fuel_types = car['fuel_type'].unique()

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           fuel_types=fuel_types)

@app.route('/predict',methods=['POST'])
def predict():

    name = request.form.get('name')
    company = request.form.get('company')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kms_driven'))
    fuel_type = request.form.get('fuel_type')

    data = pd.DataFrame([[name,company,year,kms_driven,fuel_type]],
                        columns=['name','company','year','kms_driven','fuel_type'])

    prediction = model.predict(data)

    return str(int(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)