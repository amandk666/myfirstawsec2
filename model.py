import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#from google.colab import drive
#pip install imblearn
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
import pickle

# Load the data set from google drive

def load_data():
  #drive.mount('/content/drive')
  df = pd.read_csv('bank-full.csv')
  print(df.head())
  print(df.shape)
  return df

# Performing data cleaning, missing value treatment, outlier treatment on it.
def data_preprocessing():
  df = load_data()
  # Check for any missing values
  print(df.isnull().sum())
  if df.duplicated().sum() > 1:
    df = df.drop_duplicates()
  print(df.info())
  # Since there are few Categorical variable will change them to numeric using one hot encoding except for y column
  num_data = pd.get_dummies(df.drop('y', axis  =1 ))
  y = df['y'].replace('no',0).replace('yes',1)
  # Since there are 16 independent variables will try to check the correlation with the target variable
  print(df.corr())
  # As per the observation there are not much correlation for the target variable with independent variables, so will use feature selection using
  # ExtraTreeClassifier
  model = ExtraTreesClassifier()
  model.fit(num_data, y)
  print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
  #plot graph of feature importances for better visualization
  feat_importances = pd.Series(model.feature_importances_, index=num_data.columns)
  feat_importances.nlargest(7).plot(kind='barh')
  plt.show()
  print(feat_importances.nlargest(7))

  # we considered 7 top features which has impact on target variable
  X_new = num_data[['duration', 'day', 'age', 'balance','pdays','poutcome_success','campaign']]
  df["y"].value_counts()
  # Since there are imbalance class, will use imblearn under sampling to treat this in the model training part 
  return X_new, y

def model_train_test():
  X, y = data_preprocessing()
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 31)
  rus = RandomUnderSampler()
  X_rus, y_rus = rus.fit_resample(X_train, y_train)
  lr = LogisticRegression()
  lr.fit(X_rus, y_rus)
  y_pred = lr.predict(X_test)
  accML = pd.DataFrame(classification_report(y_test, y_pred, output_dict = True)).accuracy[0].round(2)
  accNN = neural_mod(X_rus, y_rus)
  pickle.dump(lr, open('model.pkl','wb'))

def neural_mod(X_rus, y_rus):
  model = Sequential()
  model.add(Dense(12, input_dim=7, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # compile the keras model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit the keras model on the dataset
  model.fit(X_rus, y_rus, epochs=150, batch_size=10)
  # evaluate the keras model
  _, accuracy = model.evaluate(X_rus, y_rus)
  print('Accuracy: %.2f' % (accuracy*100))
  return (accuracy*100)

def prediction(data):
    #mod = model_train_test()
    print("inside model",data)
    model = pickle.load(open('pickle/model.pkl','rb'))
    print("here in model")
    predData = model.predict(data)
    return predData
