import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectPercentile, chi2

# Define the pipeline creation function
def get_pipeline(percentile: int, **hyperparams) -> Pipeline:
    '''Pipeline we use for our model.'''
    
    # Print statement for initializing pipeline 
    print("Creating pipeline with hyperparameters:", hyperparams)

    # Print statement for the current percent of features selected using chi2
    print(f'Selecting best {percentile} percent of features.')

    # Create pipeline - first use feature selection then fit SGDCLassifier with optimized hyperparams
    selector = SelectPercentile(chi2, percentile=percentile)
    pipeline = Pipeline([
        ('selector', selector),
        ('classifier', SGDClassifier(**hyperparams))
    ])
    print("Pipeline created.")
    return pipeline