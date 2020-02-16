import pickle

import flask
import numpy as np
import pandas as pd

app = flask.Flask(__name__)

app.config['DEBUG'] = True
@app.route('/', methods=['GET', 'POST'])
# Routes API to localhost

def post_feature():
    petal_width = flask.request.args.get('petal_width')
    petal_length = flask.request.args.get('petal_length')
    # Get features for model from URL as queries

    X = pd.DataFrame({'petal length (cm)': petal_length, 'petal width (cm)': petal_width}, index=[0])
    # Formats features as DataFrame for prediction

    prediction = model.predict(X)

    return np.array2string(prediction[0])

if __name__ == '__main__':
    model = pickle.load(open('model.pickle', 'rb'))
    # Deserializes model

    app.run(debug=True)