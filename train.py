import os
import pickle

from keras.models import Sequential
from keras.layers import Dropout, Conv1D, Flatten, Dense, Embedding
from keras.preprocessing.text import Tokenizer

from utils import read_data, preprocess_df, split_data, tokenize_data
from constants import MAX_WORDS


def train():
    df = read_data(os.path.join('data', 'IMDB Dataset.csv'), train=True)
    df = preprocess_df(df)
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(df)

    token = Tokenizer(lower=False)
    token.fit_on_texts(x_train)

    # Saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x_train_dl = tokenize_data(token, x_train, max_words=MAX_WORDS)
    x_val_dl = tokenize_data(token, x_val, max_words=MAX_WORDS)
    x_test_dl = tokenize_data(token, x_test, max_words=MAX_WORDS)

    total_words = len(token.word_index) + 1
    model = build_model(total_words)

    model.fit(x_train_dl, y_train, validation_data=(x_val_dl, y_val), epochs=1, batch_size=64)

    if not os.path.exists('models'):
        os.mkdir('models')
    model.save(os.path.join('models', 'sentiment_analysis_model.h5'))

    score = model.evaluate(x_test_dl, y_test)
    print(score[1])
    return model


def build_model(total_words):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=32, input_length=MAX_WORDS))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    train()
