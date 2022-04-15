import os
import pickle

from flask import request
from flask import jsonify
from flask import Flask, render_template
from keras.models import load_model

from train import train
from utils import read_data, preprocess_df, tokenize_data
from constants import MAX_WORDS

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    text = read_data(text)
    text = preprocess_df(text)
    text = tokenize_data(tokenizer, text['review'], max_words=MAX_WORDS)
    score = model.predict(text)

    if score > 0.5:
        label = 'This sentence is positive'
    elif score == 0.5:
        label = 'This sentence is neutral'
    else:
        label = 'This sentence is negative'
    return render_template('index.html', variable=label)


if __name__ == "__main__":
    if not os.path.exists(os.path.join('models', 'sentiment_analysis_model.h5')):
        train()
    model = load_model(os.path.join('models', 'sentiment_analysis_model.h5'))
    # loading
    with open(os.path.join('tokenizer', 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    app.run(port='8088', threaded=False)
