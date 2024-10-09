# print("Ritu")

from flask import Flask, render_template,jsonify,request,redirect,url_for,jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

lr_model = pickle.load(open("RegressorModel.pkl",'rb')) 

df = pd.read_csv('static/cleaned_car.csv')


@app.route('/')
def index():
    companies = df.company.sort_values().unique().tolist()
    models = df.name.sort_values().unique().tolist()
    years = df.year.sort_values().unique()
    fuels = df.fuel_type.sort_values().unique()
    # print('Ritu model',models,'type',type(models),)
    return render_template('index.html',companies=companies,models=models,years = years,fuels=fuels)

@app.route('/predict/',methods = ['POST'])
def predict():
    if request.method =='POST':
        company = request.form['company']
        model = request.form['model']
        year = int(request.form['year'])
        fuel = request.form['fuel']
        kms = int(request.form['kms'])


        data = pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([model, company, year, kms, fuel]).reshape(1,5))
        # print('data',data)
        output = lr_model.predict(data)
        response = {
            'result': round(output[0], 2)  # Round the output to 2 decimal places
        }
        
        return jsonify(response)
    return redirect(url_for('index'))


if __name__ == "__main__":
   app.run(debug=True)