import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn

def main():
    df = pd.read_table("../data/train.tsv")
    df = format_df(df)

def format_df(df):
    import json
    from sklearn.preprocessing import MultiLabelBinarizer

    # we have to remove the records with only NaN in the description
    df = df[-df['Product Name'].isnull()]

    df['label'] = df.tag.apply(lambda x: json.loads(x))
    blabels = MultiLabelBinarizer().fit_transform(df['label'])
    df['blabel'] = pd.Series(list(blabels))

    df = df[['item_id', 'Product Long Description', 'blabel']]

    return df



from bs4 import BeautifulSoup

soups = [BeautifulSoup(item, "lxml").get_text(separator="\n") for item in df['Product Long Description']]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(soups)

# Perform actual TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)

# PCA JUST FOR FUN
from sklearn.decomposition import PCA
X = PCA(n_components=2).fit_transform(X_train_tfidf.toarray())
plt.scatter(X[:,0],X[:,1])

# multilabel classification with random forest
# One might choose a different classifier if needed.
from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1)
y = np.array(df['blabel'].tolist())
# print(X_train_tfidf.shape,\
# y.shape) #, dtype="|S6")

# Training and validation sets
X_train, X_test = X_train_tfidf[:8000], X_train_tfidf[8000:]
y_train, y_test = y[:8000], y[8000:]

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

f1_score(y_test, y_pred)

clf.fit(X_train_tfidf, y)




def f1_score(labels, predictions):
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
                if y_pred[i,j] == 1:
                    fp += 1
                else:
                    fn += 1
    accuracy = (tp + tn)/len(labels)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*precision * recall / (precision + recall)
    return accuracy, precision, recall, f1

if __name__ == "__main__": main()
