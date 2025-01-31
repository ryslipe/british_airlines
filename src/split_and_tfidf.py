import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Tuple


from src.paths import TRANSFORMED_DATA_DIR

def split_and_vect(df: pd.DataFrame, X_split: str, y_split: str) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series, TfidfVectorizer]:
    # split x and y
    X = df[X_split]
    y = df[y_split]

    # train test split
    X_train_, X_test_, y_train_, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range= (1, 2), stop_words='english')

    # fit transform to training data
    X_train_vec = tfidf_vectorizer.fit_transform(X_train_)

    # transform test data
    X_test_vec = tfidf_vectorizer.transform(X_test_)

    return X_train_vec, X_test_vec, y_train_, y_test, tfidf_vectorizer
