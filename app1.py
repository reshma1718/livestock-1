import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app1 = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))

@app1.route('/')
def home():
    return render_template('index1.html')

@app1.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model1.predict(final_features)
    
    output = round(prediction[0], 2)

    return render_template('index1.html', prediction_text='predicted buffalow census to be million {}'.format(output))

if __name__ == "__main__":
    app1.run(debug=True)


    