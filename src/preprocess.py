import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def main():

    test = False
    test = True
    if test:
        df = pd.read_table("../data/train.tsv")
        df_test = pd.read_table("../data/test.tsv")
        df_test = df_test[-df_test['Product Long Description'].isnull()]
        df_test = df_test.iloc[:1000]
    else:
        df = pd.read_table("../data/train.tsv")
        # Uncomment to reduce data for testing purposes
        df = df.iloc[:1000]

    df, blabels, mlb = format_df(df)

    # Extract a tfidf sparse matrix
    prep_done = False
    if not prep_done:
        if test:
            soups = [BeautifulSoup(item, "lxml").get_text(separator="\n")
                     for item in df['Product Long Description']]
            # X_tfidf = tf_idf(df['Product Long Description'])
            count_vec = joblib.load('cnt_vec.pkl')
            X_tfidf_test = count_vec.fit(soups)
        else: # train
            X_tfidf, count_vec = tf_idf(df['Product Long Description'])
            joblib.dump(count_vec, 'cnt_vec.pkl')
    else:
        pass # In case we have saved the tf-idf
    # print([np.array(x).reshape((1,-1)).shape for x in df['blabel']][:2])
    # exit("DEBUG")
    # y = df['blabel'].values
    # y = [np.array(x) for x in df['blabel']]
    if test:
        # clf = train_clf(X_tfidf, y)
        clf = joblib.load('clf.pkl')
        predictions = clf.predict(X_tfidf_test)

    else:
        y = blabels

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_tfidf, y, test_size=.2)

        clf = train_clf(X_train, y_train)
        predictions = clf.predict(X_valid)
        scores = all_scores(y_valid, predictions)
        print("\n All scores on training/validation sets: \n{}".format(scores))
        joblib.dump(clf, 'clf.pkl')

    shelves = mlb.inverse_transform(predictions)
    # Tags processing
    straight_list = straighten_tags(df['tag'])
    straight_list = [int(s) for s in straight_list]
    straight_list = np.array(straight_list)
    s_tags = pd.Series(straight_list)
    mode = s_tags.mode()

    final_shelves = []
    for x in shelves:
        if x != ():
            final_shelves.append(list(x))
        else:
            final_shelves.append(list(mode))
    df['shelves'] = pd.Series(final_shelves)

    print(df.shelves.iloc[:10])

    df[['item_id', 'shelves']].to_csv(
        "tags.tsv", sep="\t", header=["item_id", "tag"], index=False)



def format_df(df):
    import json
    from sklearn.preprocessing import MultiLabelBinarizer

    # we have to remove the records with only NaN in the description
    df = df[-df['Product Name'].isnull()]

    df['label'] = df['tag'].apply(lambda x: json.loads(x))
    mlb = MultiLabelBinarizer()
    blabels = mlb.fit_transform(df['label'])
    # print(blabels.shape)
    df['blabel'] = pd.Series(list(blabels))

    df = df[['item_id', 'Product Long Description', 'tag', 'blabel']]

    return df, blabels, mlb


def tf_idf(pandas_series, test = False):

    # print("{}\n{}\n{}".format(pandas_series[0], type(pandas_series), pandas_series.shape))
    soups = [BeautifulSoup(item, "lxml").get_text(separator="\n")
             for item in pandas_series]

    count_vect = CountVectorizer()
    count_vect.fit(soups)
    X_train_counts = count_vect.transform(soups)

    # Perform actual TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_train_tf = tf_transformer.transform(X_train_counts)
    """
    # PCA JUST FOR FUN
    from sklearn.decomposition import PCA
    X = PCA(n_components=2).fit_transform(X_train_tfidf.toarray())
    plt.scatter(X[:,0],X[:,1])
    """
    return X_train_tfidf, count_vect


def straighten_tags(df_tags):
    mylist = []
    for tags in df_tags:
            mylist += tags.strip(']').strip('[').split(',')
    return mylist


def train_clf(X, y, clf_type='rnd_forest'):
    """
    multilabel classification with random forest
    One might choose a different classifier if needed.
    :param X:
    :param y:
    :return: clf: trained classifier
    """
    # clf = RandomForestClassifier()
    from sklearn.multiclass import OneVsRestClassifier
    if clf_type == 'rnd_forest':
        from sklearn.ensemble import RandomForestClassifier
        myclf = RandomForestClassifier()
    elif clf_type == 'logreg':
        from sklearn.linear_model import SGDClassifier
        myclf = SGDClassifier()
    else:
        raise ValueError("No other classifier type is possible yet.")
    clf = OneVsRestClassifier(myclf, n_jobs=-1)

    clf.fit(X, y)

    # y_pred = clf.predict(X_val)

    # scores = all_scores(y_val, y_pred)

    # clf.fit(X_train_tfidf, y)

    return clf


def all_scores(labels, predictions):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(labels)):
        if (predictions[i] == labels[i]).all():
            tp += len(predictions[i][predictions[i] == 1])
            tn += len(predictions[i][predictions[i] == 0])
        else:
            for j in labels[i, predictions[i] != labels[i]]:
                if predictions[i,j] == 1:
                    fp += 1
                else:
                    fn += 1
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


if __name__ == "__main__": main()

