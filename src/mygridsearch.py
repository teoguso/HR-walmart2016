
"""
==========================================================
Sample pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached and reused for the document
classification example.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.

Here is a sample output of a run on a quad-core machine::

  Loading 20 newsgroups dataset for categories:
  ['alt.atheism', 'talk.religion.misc']
  1427 documents
  2 categories

  Performing grid search...
  pipeline: ['vect', 'tfidf', 'clf']
  parameters:
  {'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
   'clf__n_iter': (10, 50, 80),
   'clf__penalty': ('l2', 'elasticnet'),
   'tfidf__use_idf': (True, False),
   'vect__max_n': (1, 2),
   'vect__max_df': (0.5, 0.75, 1.0),
   'vect__max_features': (None, 5000, 10000, 50000)}
  done in 1737.030s

  Best score: 0.940
  Best parameters set:
      clf__alpha: 9.9999999999999995e-07
      clf__n_iter: 50
      clf__penalty: 'elasticnet'
      tfidf__use_idf: True
      vect__max_n: 2
      vect__max_df: 0.75
      vect__max_features: 50000

"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

# from __future__ import print_function

from pprint import pprint
from time import time
import logging

# from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd
from sklearn.externals import joblib
from bs4 import BeautifulSoup
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# print(__doc__)
from sklearn.datasets import fetch_20newsgroups

import json
from sklearn.model_selection import train_test_split


def format_df(df):
    import json

    # we have to remove the records with only NaN in the description
    df = df[-df['Product Name'].isnull()]

    df['label'] = df['tag'].apply(lambda x: np.array(eval(x)))
    # print([(type(x), x) for x in df.label[:9].values])
    # print(df.label.values[:10].shape)
    # exit("DEBUG")
    # df['label'] = df['tag'].apply(lambda x: json.loads(x))
    mlb = MultiLabelBinarizer()
    blabels = mlb.fit_transform(df['label'])
    # print(blabels.shape)
    df['blabel'] = pd.Series(list(blabels))

    df = df[['item_id', 'Product Long Description', 'tag', 'blabel']]

    return df, blabels, mlb


def main():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

    # ========================================= #
    ###############################################################################
    # Load some categories from the training set
    # categories = [
    #     'alt.atheism',
    #     'talk.religion.misc',
    # ]
    # Uncomment the following to do the analysis on all the categories
    #categories = None
    # print("Loading 20 newsgroups dataset for categories:")
    # print(categories)
    # data = fetch_20newsgroups(subset='train', categories=categories)
    # print("%d documents" % len(data.filenames))
    # print("%d categories" % len(data.target_names))
    # print(data.data[:1], data.target[:1])
    # exit("DEBUG")
    # ======================================================= #

    test = True
    test = False
    if test:
        df = pd.read_table("../data/train.tsv")
        df_test = pd.read_table("../data/test.tsv")
        df_test = df_test[-df_test['Product Long Description'].isnull()]
        df_test = df_test.iloc[:1000]
    else:
        df = pd.read_table("../data/train.tsv")
        # Uncomment to reduce data for testing purposes
        # df = df.iloc[:1000]

    df, blabels, mlb = format_df(df)

    soups = [BeautifulSoup(item, "lxml").get_text(separator="\n")
             for item in df['Product Long Description']]

    X, y = soups, blabels # .ravel()
    print(np.array(X).shape, np.array(y).shape)
    # print(np.array(X).shape, np.array(y).shape)
    # exit("DEBUG")
    # print()

    ###############################################################################
    # define a pipeline combining a text feature extractor with a simple
    # classifier
    myclf = SGDClassifier()
    clf = OneVsRestClassifier(myclf, n_jobs=-1)
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(myclf, n_jobs=-1)),
    ])
    # ('clf', SGDClassifier()),

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }

    parameters_better = {
        'vect__max_df': (1.0, 1.25),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 2), (1, 3) ),  # unigrams or bigrams or trigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__estimator__alpha': (1e-6, ),
        'clf__estimator__penalty': ('elasticnet', ),
        #'clf__n_iter': (10, 50, 80),
    }
    # 'clf__estimator__alpha': ((0.00001),), # 0.000001),
    # 'vect__ngram_range': ((1, 1),), # (1, 2)),  # unigrams or bigrams
    # 'vect__max_df': (0.5,), #, 0.75, 1.0),
    # 'estimator__clf__alpha': ((0.00001),), # 0.000001),
    # 'estimator__clf__penalty': ('l2',), #, 'elasticnet'),

    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    parameters = parameters_better
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1_samples')

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    # grid_search.fit(data.data, data.target)
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


    joblib.dump(grid_search.best_estimator_, "best.pkl")
    joblib.dump(pipeline, "pipeline.pkl")


if __name__ == "__main__": main()

