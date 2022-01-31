import flask
import os

from flask import Flask, url_for, render_template, request, redirect, session
#from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

# Create the application.
app = flask.Flask(__name__)

reviews = pd.read_csv('dataset/bank-full.csv')

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        incData = []
        dur = request.form['duration']
        day = request.form['day']
        age = request.form['age']
        bal = request.form['balance']
        pday =request.form['pdays']
        pout =request.form['poutcome_success']
        cam = request.form['campaign']
        input = [dur, day, age, bal, pday, pout, cam]
        for item in input:
          items = item.split(',')
          incData.append(items)
        print(incData)
        try:
            data = pd.DataFrame(columns=['duration','day','age','balance','pdays','poutcome_success','campaign'])
            incData = np.transpose(incData)
            data = pd.DataFrame(incData)
            print(data)
            
            if not data.empty:
                print("here",data)
                print("inside model",data)
                model = pickle.load(open('pickle/model.pkl','rb'))
                print("here in model")
                result = model.predict(data)
                print(result)
                
                return  render_template('view.html',tables= result, titles = ['prediction'])
               
            else:
                print(data)
                return render_template('invalid.html')

        except:
            print(data)
            return render_template('invalid.html')


if __name__ == '__main__':
    app.debug=True
    
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '5000'))
    #app.run()
    
