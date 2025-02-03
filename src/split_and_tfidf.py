import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Tuple


from src.paths import TRANSFORMED_DATA_DIR

def split_and_vect(df: pd.DataFrame, X_split: str, y_split: str) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series, TfidfVectorizer]:
    # split x and y
    X = df[X_split]
    y = df[y_split]

    # train test split
    # stratified shuffle split
    # sss.split(X, y) performs the stratified splitting 
    # returns indices for the training and testing sets keeping same class distribution as in y
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # perform the stratified split
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range= (1, 2), stop_words='english')

    # fit transform to training data
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)

    # transform test data
    X_test_vec = tfidf_vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, tfidf_vectorizer
