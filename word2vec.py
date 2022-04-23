import gensim.downloader as api
import pandas as pd


def load_embedding():
    return api.load('word2vec-google-news-300')


def find_similarity(word, wv):
    # wv = load_embedding()
    try:
        wv[word]
    except:
        raise Exception('Word ' + word + ' does not exist in dataset. Please try a different one.')
    similarities = wv.most_similar(positive=[word], topn=5)
    df = pd.DataFrame(similarities, columns=['Closest words', 'Similarity scores'])
    # df = pd.DataFrame(similarities, columns=['Closest words', 'Similarity scores']).to_html(index=False, classes='data', header="true")
    return df


if __name__ == '__main__':
    print('Loading embedding...')
    embedding = load_embedding()
    print('Success!')
    sim = find_similarity('king', embedding)
