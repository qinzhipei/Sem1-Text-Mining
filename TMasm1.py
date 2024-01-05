# -*- coding: utf-8 -*-
'''import packages'''
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import  TfidfVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn import metrics
from stopwordsiso import stopwords

'''import data'''
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = twenty_test.data

'''Feature Extractors'''
feature_extractors = {
    "Counts": CountVectorizer(),
    "TF": TfidfVectorizer(use_idf=False),
    "TF-IDF": TfidfVectorizer()
}

'''classifiers'''
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

''' Comparing the performance of different feature extractors and classifiers '''
for feature_name, feature_extractor in feature_extractors.items():
    for classifier_name, classifier in classifiers.items():
        t0 = time()
        print(f"Training {classifier_name} classifier with {feature_name} feature...")
        pipeline = Pipeline([
            ('vectorizer', feature_extractor),
            ('classifier', classifier)
        ])

        pipeline.fit(twenty_train.data, twenty_train.target)
        y_pred = pipeline.predict(twenty_test.data)

        precision = precision_score(twenty_test.target, y_pred, average='weighted')
        recall = recall_score(twenty_test.target, y_pred, average='weighted')
        f1 = f1_score(twenty_test.target, y_pred, average='weighted')
        t1 = time()
        print(t1-t0)
        print(f"Precision for {classifier_name} classifier with {feature_name} feature: {precision:.2f}")
        print(f"Recall for {classifier_name} classifier with {feature_name} feature: {recall:.2f}")
        print(f"F1 score for {classifier_name} classifier with {feature_name} feature: {f1:.2f}\n")


'''Parameters'''
Lowercase = [True,False]
ngram_range = [(1, 1),(1, 2),(1, 3),(1, 4),(2, 2),(2, 3)]
english_stopwords = stopwords(["en"])
stop_words = [None,'english',list(english_stopwords)]
max_features = [None,500,2000,5000]

'''Parameters Evaluation'''
#Lowercase
for i in range(len(Lowercase)):
    t0 = time()
    text_clf = Pipeline([
     ('vect', CountVectorizer(lowercase=Lowercase[i])),
     ('tfidf', TfidfTransformer()),
     ('clf', SVC(kernel='linear')),
     ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    t1 = time()
    print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names))
    print(t1-t0)

#stop_words
for i in range(len(stop_words)):
    t0 = time()
    text_clf = Pipeline([
     ('vect', CountVectorizer(stop_words=stop_words[i])),
     ('tfidf', TfidfTransformer()),
     ('clf', SVC(kernel='linear')),
     ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    t1 = time()
    print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names))
    print(t1-t0)
    
#ngram_range
for i in range(len(ngram_range)):
    t0 = time()
    text_clf = Pipeline([
     ('vect', CountVectorizer(ngram_range=ngram_range[i])),
     ('tfidf', TfidfTransformer()),
     ('clf', SVC(kernel='linear')),
     ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    t1 = time()
    print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names))
    print(t1-t0)

#max_features
for i in range(len(max_features)):
    t0 = time()
    text_clf = Pipeline([
     ('vect', CountVectorizer(max_features=max_features[i])),
     ('tfidf', TfidfTransformer()),
     ('clf', SVC(kernel='linear')),
     ])
    text_clf.fit(twenty_train.data, twenty_train.target)
    predicted = text_clf.predict(docs_test)
    t1 = time()
    print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names))
    print(t1-t0)











