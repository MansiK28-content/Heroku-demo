import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    area = float(request.form.get('area'))
    bedroom = int(request.form.get('bedroom'))
    bathrooms = int(request.form.get('bathrooms'))
    stories = int(request.form.get('stories'))
    guestroom = int(request.form.get('guestroom'))
    basement = int(request.form.get('basement'))
    parking = int(request.form.get('parking'))
    areaperbedroom = float(request.form.get('areaperbedroom'))
    bbratio = float(request.form.get('bbratio'))
    prediction = model.predict(
        [[area, bedroom, bathrooms, stories, guestroom, basement, parking, areaperbedroom, bbratio]])
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The house price predicted is Rupees {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
