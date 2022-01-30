import flask
import os

from flask import Flask, url_for, render_template, request, redirect, session
#from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
import model

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
        name = request.form['username']
        try:
            d = [{'duration':0, 'day':1, 'age':23, 'balance':2,'pdays':1,'poutcome_success':0,'campaign':1}]
            data = pd.DataFrame(d)
            print(data)
            
            if not data.empty:
                print("here",data)
                result = model.prediction(data)
                print(result)
                return  render_template('view.html',tables=[result[0]], titles = ['prediction'])
               
            else:
                print(name)
                return render_template('invalid.html')

        except:
            print(name)
            return render_template('invalid.html')


if __name__ == '__main__':
    app.debug=True
    
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '5000'))
    #app.run()
    