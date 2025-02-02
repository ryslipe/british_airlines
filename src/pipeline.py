import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectPercentile, chi2

# create pipeline
def get_pipeline(percentile: int, **hyperparams) -> Pipeline:
    '''Pipeline we use for our model.'''
    # print statement
    print("Creating pipeline with hyperparameters:", hyperparams)
    # print statement for feature selection
    print(f'Selecting best {percentile} percent of features.')
    # create pipeline
    selector = SelectPercentile(chi2, percentile=percentile)
    pipeline = make_pipeline(
        selector,
        SGDClassifier(**hyperparams)
    )
    print("Pipeline created:")
    return pipeline