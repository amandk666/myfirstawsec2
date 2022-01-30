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
        dur = list(request.form['duration'])
        day = list(request.form['day'])
        age = list(request.form['age'])
        bal = list(request.form['balance'])
        pday = list(request.form['pdays'])
        pout = list(request.form['poutcome_success'])
        cam = list(request.form['campaign'])
        print(dur) 
        try:
            data = pd.DataFrame(columns=['duration','day','age','balance','pdays','poutcome_success','campaign'])
            lstD = []
            lstD.append(dur)
            lstD.append(day)
            lstD.append(age)
            lstD.append(bal)
            lstD.append(pday)
            lstD.append(pout)
            lstD.append(cam)
            print(lstD)
            lstD = np.transpose(lstD)
            data = pd.DataFrame(lstD)
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
    
