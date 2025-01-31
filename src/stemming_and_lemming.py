import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk

# stemmer function
def stem_reviews(reviews: pd.Series) -> pd.Series:
    ps = PorterStemmer()
    stemmed_reviews = reviews.apply(lambda review: ' '.join([ps.stem(word) for word in word_tokenize(review)]))
    return stemmed_reviews


# lemmetizer pos
def get_wordnet_pos(tag: str) -> str:
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# lemmatizer function
def lemmatize_reviews(reviews: pd.Series) -> pd.Series:
    lemmatizer = WordNetLemmatizer()
    lemmatized_reviews = []
    for review in reviews:
        tokens = word_tokenize(review)
        pos_tags = pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
        lemmatized_reviews.append(' '.join(lemmatized))
    return pd.Series(lemmatized_reviews)

