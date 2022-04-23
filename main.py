import os
import pickle

from flask import request
from flask import jsonify
from flask import Flask, render_template
from keras.models import load_model

from train import train
from word2vec import find_similarity, load_embedding
from utils import read_data, preprocess_df, tokenize_data
from constants import MAX_WORDS

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    request_dict = request.form.to_dict()
    sentiment = False
    word2vec = False

    if 'text' in request_dict.keys():
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
        sentiment = True

    if 'word2vec' in request_dict.keys():
        word = request.form['word2vec']
        print(word)
        similar_df = find_similarity(word, embedding)
        print(similar_df)
        column_names = similar_df.columns.values
        row_data = list(similar_df.values.tolist())
        link_column = "Patient ID"
        word2vec = True

    if sentiment and word2vec:
        return render_template('index.html', variable=label, column_names=column_names, row_data=row_data, link_column=link_column, zip=zip)
    elif sentiment:
        return render_template('index.html', variable=label)
    elif word2vec:
        return render_template('index.html', column_names=column_names, row_data=row_data, link_column=link_column, zip=zip)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    print('Loading embedding...')
    embedding = load_embedding()
    print('Success!')

    if not os.path.exists(os.path.join('models', 'sentiment_analysis_model.h5')):
        train()
    model = load_model(os.path.join('models', 'sentiment_analysis_model.h5'))
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    app.run(port='8088', threaded=False)
