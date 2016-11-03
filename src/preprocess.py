import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn


def main():

    test = False
    if test:
        df = pd.read_table("../data/train.tsv")
        df_test = pd.read_table("../data/test.tsv")
    else:
        df = pd.read_table("../data/train.tsv")
        df, blabels, mlb = format_df(df)

    # Uncomment to reduce data for testing purposes
    # df = df.iloc[:1000]


    # Extract a tfidf sparse matrix
    prep_done = False
    if not prep_done:
        if test:
            X_tfidf = tf_idf(df['Product Long Description'])
            X_tfidf_test = tf_idf(df_test['Product Long Description'])
        else:
            X_tfidf = tf_idf(df['Product Long Description'])
    else:
        pass
    # print([np.array(x).reshape((1,-1)).shape for x in df['blabel']][:2])
    # exit("DEBUG")
    # y = df['blabel'].values
    # y = [np.array(x) for x in df['blabel']]
    if not test:
        y = blabels

        predictions, scores = classification(X_tfidf, y)
        print(scores)

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


def tf_idf(pandas_series):
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    soups = [BeautifulSoup(item, "lxml").get_text(separator="\n")
             for item in pandas_series]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(soups)

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

    return X_train_tfidf

def straighten_tags(df_tags):
    mylist = []
    for tags in df_tags:
            mylist += tags.strip(']').strip('[').split(',')
    return mylist

def classification(X,y,valid_ratio=.2):
    # multilabel classification with random forest
    # One might choose a different classifier if needed.
    from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier()
    from sklearn.multiclass import OneVsRestClassifier
    clf = OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1)
    # print(X_train_tfidf.shape,\
    # y.shape) #, dtype="|S6")

    # Training and validation sets
    len_valid = int(len(y)*(1-valid_ratio))
    X_train, X_val = X[:len_valid], X[len_valid:]
    y_train, y_val = y[:len_valid], y[len_valid:]

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    scores = all_scores(y_val, y_pred)

    # clf.fit(X_train_tfidf, y)

    return y_pred, scores


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

