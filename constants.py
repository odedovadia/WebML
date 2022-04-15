import nltk
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
tokenizer = ToktokTokenizer()

MAX_WORDS = 500
