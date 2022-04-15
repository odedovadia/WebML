import re

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing import sequence

from constants import *


def read_data(input_text: str, train: bool = False):
    if train:
        return pd.read_csv(input_text)
    return pd.DataFrame(data={'review': [input_text], 'sentiment': ['Unknown']})


# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing the noisy text
def clean_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text)
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def preprocess_df(input_df: pd.DataFrame):
    tqdm.pandas()
    input_df['review'] = input_df['review'].progress_apply(clean_text)
    input_df['review'] = input_df['review'].progress_apply(remove_special_characters)
    input_df['review'] = input_df['review'].progress_apply(remove_stopwords)
    # input_df['review'] = input_df['review'].progress_apply(simple_stemmer)
    return input_df


def split_data(df):
    x = df['review']
    y = df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    y_val = lb.transform(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val


def tokenize_data(token, x, max_words):
    x_dl = token.texts_to_sequences(x)
    x_dl = sequence.pad_sequences(x_dl, maxlen=max_words)
    return x_dl


if __name__ == '__main__':
    a = read_data('Hello, how are you? <br /><br />')
    b = preprocess_df(a)
