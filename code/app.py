from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model=pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_data():
    x=[[x for x in request.form.values()]]
    print(x)
    x=np.array(x)
    print(x.shape)
    prediction=model.predict(x)
    print(prediction[0])
    return render_template('submit.html', prediction_result=str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
