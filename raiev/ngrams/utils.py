from nltk import sent_tokenize, everygrams
from nltk.corpus import stopwords
import re


def preprocess_sentence(sentence):
    # lowercase sentence
    sentence = sentence.lower()
    # remove ending punctuation from sentence
    sentence = sentence[:-1] if sentence[-1] in [';', '?', '!', '.'] else sentence
    # replace multiple dashes with a single dash
    sentence = re.sub(r'\-+', '-', sentence)
    # standardize single/double quotes and remove double instances of single quotes
    sentence = sentence.replace('`', "'").replace('“', '"').replace('”', '"').replace("''", '')
    # replace parentheses and double quotes with a space
    sentence = re.sub('\(|\)|\"|\[|\]', ' ', sentence)
    # remove urls
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    sentence = re.sub(url_regex, '', sentence)
    # replace various punctuation with a space
    sentence = re.sub(",\ |\.,\ |;\ |-\ |:\ |'\ |•|►", ' ', sentence)
    sentence = re.sub("\ '", ' ', sentence)
    # replace multiple spaces with a single space
    sentence = re.sub(r'\ +', ' ', sentence)
    return sentence.split()
    
def extract_ngrams(df, id_col, text_col, ngram_col='Ngram', minN=2, maxN=2):
    """
    Given text and id, generate a list of ngrams for each id  supplied.
    df (pandas dataframe) - dataframe containing id and text
    id_col (str) - id column
    text_col (str) - text column
    ngram_col (str) - optional, how to name the ngram column produced
    minN (int) - optional, minimum N value
    maxN (int) - optional, maximum N value
    """
    text = df[[id_col, text_col]].copy()
    # Split each text into sentences 
    text['sentence'] = text[text_col].apply(lambda x: sent_tokenize(re.sub(r'\.+', '.', x)))
    text = text.explode('sentence')
    # Standard preprocessing steps for each sentence
    text['sentence'] = text['sentence'].apply(lambda x: preprocess_sentence(x))
    # Get ngrams
    text[ngram_col] = text['sentence'].apply(lambda x: list(everygrams(x, minN, maxN)))
    text = text.explode(ngram_col).dropna(subset=ngram_col)
    # Remove ngrams that contain stop words
    stop_words = stopwords.words("english")
    text = text[text[ngram_col].apply(lambda x: not any([word in stop_words for word in x]))]
    text[ngram_col] = text[ngram_col].apply(lambda x: ' '.join(x))
    text = text.groupby(id_col)[ngram_col].apply(lambda x: list(x))
    return text
