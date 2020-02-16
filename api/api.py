import pickle

import flask
import numpy as np
import pandas as pd
from flask_restful import Api

app = flask.Flask(__name__)
api = Api(app)

todos = {}

app.config['DEBUG'] = True
@app.route('/', methods=['GET', 'POST'])
def post_feature():
    petal_width = flask.request.args.get('petal_width')
    petal_length = flask.request.args.get('petal_length')
    X = pd.DataFrame({'petal length (cm)': petal_length, 'petal width (cm)': petal_width}, index=[0])
    prediction = model.predict(X)

    return np.array2string(prediction)

if __name__ == '__main__':
    model = pickle.load(open('model.pickle', 'rb'))
    app.run(debug=True)


