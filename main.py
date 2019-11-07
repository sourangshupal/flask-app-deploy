from flask import Flask, request  #import flask library & request module
from flask_cors import CORS #for cloud deployment
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
import pandas as pd
import os
from requests import Response

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)
@app.route("/",methods=['POST'])
def home():
    print("Welcome to Salary prediction!")
    return "Welcome to Salary Prediction!...Access '/predict' to predict your salary"
    
@app.route("/predict",methods=['POST'])
def predict():
    inp=request.json
    print(inp['inp'])
    inp_arr=float(inp['inp']) #a 2D array (1,1)
    loaded_model = pickle.load(open('salary_model.sav','rb'))
    return str(loaded_model.predict([[inp_arr]]))

@app.route("/train",methods=['POST'])
def train():
    
    with open('data/data.json', 'w', encoding='utf-8') as f:
        json.dump(request.json['inp'], f, ensure_ascii=False, indent=4)
    df1 = pd.read_json('data/data.json')
    #print(df1.iloc[:])
    
    df1.to_csv('data/input.csv', index=None, header=True)
    
    df2 = pd.read_csv(r'data/input.csv')  
    print(df2.iloc[:])
    #read in the X and y values
    X = df2.iloc[0:,0:1] #read in X as a 2D array of values shape (n,1)
    y = df2.iloc[:,1] #read in y as 1D array of values
    
    #split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
    
    #fit a simple linear regression to the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    #save the model
    filename='salary_model.sav'
    pickle.dump(regressor,open(filename,'wb'))
    
    #loaded_model = pickle.load(open(filename,'rb'))
    
    return "Model trained successfully"

port = int(os.getenv("PORT"))
if __name__ == '__main__':
#checking if the __name__ is the name of main module
    app.run('0.0.0.0',port=port) #Yes it is..then run the server application