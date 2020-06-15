import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
Tf = TfidfVectorizer(max_features=5000)

@app.route('/')
def home():
    return render_template('home_html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector = Tf.transform(data).toarray()
        my_prediction = model.predict(vector)
        return render_template('result.html',prediction = my_prediction)
    
if __name__ == '__main__':
    app.run(debug= True)
    
    
        
