# -*- coding: utf-8 -*-

#! pip install hazm
#! pip install pandas
#! pip install sklearn
#! pip install numpy

"""Imports"""

import pandas as pd

train_ds_path = 'train.csv'
print('Reading train dataset from', train_ds_path)
dataset = pd.read_csv(train_ds_path, index_col=0)

"""# Utils

A function for printing Progress Bar
Thanks to [Greenstick](https://stackoverflow.com/a/34325723)
"""

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = ' ')
    # Print New Line on Complete
    if iteration == total: 
        print()

"""# **Some visualization on dataset**"""

dataset.head(3)

TEXT_COL_NAME = 'Text'
CAT_COL_NAME = 'Category'
def print_row(index):
    print('row ', index)
    print(CAT_COL_NAME + ':', dataset.at[index, CAT_COL_NAME])
    print(TEXT_COL_NAME + ':', dataset.at[index, TEXT_COL_NAME])
print('Dataset Size: ', len(dataset))
#print_row(50)

def get_words_set(ds):
  _set = set()
  for index, row in ds.iterrows():
      if type(row[TEXT_COL_NAME]) is str:
          _text_arr = row[TEXT_COL_NAME].split(' ')
      else:
          _text_arr = row[TEXT_COL_NAME]
      _set.update(_text_arr)
      printProgressBar(index + 1, len(ds), prefix = 'Progress:', suffix = 'Complete', length = 50)
  return _set

def number_of_words(ds):
    return len(get_words_set(ds))

print('Extracting number of words before preprocessing...')
print('Number of words before preprocessing: ', number_of_words(dataset))

"""# **Preprocessing**

**Create a list of categories and convert categories to numbers in dataset**
"""

def get_categories(ds):
  _set = set()
  for index, row in ds.iterrows():
      _set.add(row[CAT_COL_NAME])
  return _set

print('Extracting Category names from dataset')
cats_vector = list(get_categories(dataset))

def convert_category_to_number(cat):
  return cats_vector.index(cat)

def convert_number_to_category(index):
  return cats_vector[index]

print('Converting Category names to numbers')
dataset[CAT_COL_NAME] = dataset.apply(lambda row:  convert_category_to_number(row[CAT_COL_NAME]), axis = 1)

dataset.head(5)

"""**Remove special chars and persian stop words**"""

from hazm import *
import numpy as np

normalizer = Normalizer()

def get_specific_chars():
  f = open('chars.txt', 'r')
  _tmp = f.read().split('\n')
  _tmp.append('\n')
  return _tmp
sp_chars = get_specific_chars()

def get_stop_words():
  f = open('stop-words.txt', 'r')
  _tmp = f.read().split('\n')
  for _stop in _tmp:
    if _stop.find('ی') != -1:
      _tmp.append(_stop.replace('ی', 'ي'))
  return _tmp
stop_words = get_stop_words()

def remove_consecutive_spaces(text):
  import re
  _text = re.sub(' +', ' ', text)
  while _text[0] == ' ':
    _text = _text[1:]
  while _text[len(_text)-1] == ' ':
    _text = _text[:len(_text)-1]
  return _text

def remove_specific_chars(text, sp_chars):
  _text = text
  for sp_char in sp_chars:
    _text = _text.replace(sp_char, ' ')
    _text = remove_consecutive_spaces(_text)
  return _text

def remove_stop_words(text, stop_words):
  _splited = text.split(' ')
  _res = np.array(_splited)[np.in1d(_splited, stop_words, invert = True)]
  return _res

def remove_specific_chars_and_stop_words(text, sp_chars, stop_words):
  _tmp = remove_specific_chars(text, sp_chars)
  _tmp = remove_stop_words(_tmp, stop_words)
  _tmp = ' '.join(_tmp)
  _tmp = normalizer.normalize(_tmp)
  return _tmp


def remove_specific_chars_and_stop_words_with_pb(text, sp_chars, stop_words, index, total): 
  printProgressBar(index + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
  _tmp = remove_specific_chars_and_stop_words(text, sp_chars, stop_words)
  return _tmp

print('Normalizing texts in dataset')
dataset[TEXT_COL_NAME] = dataset.apply(lambda row:  remove_specific_chars_and_stop_words_with_pb(row[TEXT_COL_NAME], sp_chars, stop_words, row.name, len(dataset[TEXT_COL_NAME])), axis = 1)

ds2 = dataset.copy()
ds3 = dataset.copy()
ds4 = dataset.copy()
ds5 = dataset.copy()

print('Extracting number of words after preprocessing...')
print('Number of words after removing special characters and stop words:', number_of_words(dataset))

"""# **splitting data and vectorization using TF-IDF**

Spliting data to X and Y
"""

X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1:].values

"""Vectorization Texts using TF-IDF"""

# Building a TF IDF matrix out of the corpus of reviews
print('Vectorizing using TF-IDF...')
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

"""Split dataset to test and train"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""# Classification"""

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
 
def calc_accuracy(y_test, y_pred):
  return accuracy_score(y_test, y_pred) * 100
 
def print_accuracy(y_test, y_pred):
  print(f"Accuracy Score -> {calc_accuracy(y_test, y_pred)}")

"""## Classification using KNN"""

#from sklearn.neighbors import KNeighborsClassifier
#KNN_clf = KNeighborsClassifier(n_neighbors=8)
#KNN_clf.fit(X_train, y_train)
#y_pred = KNN_clf.predict(X_test)

#print_accuracy(y_test, y_pred)

"""## Classification using Multinomial NB"""

print('Classification using Multinomial NB')
from sklearn.naive_bayes import MultinomialNB
MNB_clf = MultinomialNB()
MNB_clf.fit(X_train, y_train)
 
y_pred = MNB_clf.predict(X_test)
print_accuracy(y_test, y_pred)

"""## Classification using SVC"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
 
print('Classification using SVM')
SVC_model = LinearSVC(C=1, max_iter=20000, verbose=1)
SVC_clf = CalibratedClassifierCV(SVC_model, method='sigmoid', cv=5)
SVC_clf.fit(X_train, y_train)
y_pred = SVC_clf.predict(X_test)

print_accuracy(y_test, y_pred)

dataset.head(5)

"""# Test set"""

def predict_test(test_path):
  print('Read test file from', test_path)
  kaggle_test_dataset = pd.read_csv(test_path, index_col=0)
  print('Normalizing test texts', test_path)
  kaggle_test_dataset[TEXT_COL_NAME] = kaggle_test_dataset.apply(lambda row:  remove_specific_chars_and_stop_words_with_pb(row[TEXT_COL_NAME], sp_chars, stop_words, row.name, len(kaggle_test_dataset[TEXT_COL_NAME])), axis = 1)
  print('split texts to kaggle_X_test')
  kaggle_X_test = kaggle_test_dataset.iloc[:, 0].values
  print('Vectorizing kaggle_X_test')
  kaggle_X_test = vectorizer.transform(kaggle_X_test)
  print('Predicting kaggle_X_test using SVC classifier')
  kaggle_predict = SVC_clf.predict(kaggle_X_test)
  kaggle_predict_cat = ['']*len(kaggle_predict)
  print('Replacing predicted categories indexes to Category names')
  for i, cat_num in enumerate(kaggle_predict):
    kaggle_predict_cat[i] = convert_number_to_category(cat_num)
  print('Getting output from predicted categories')
  kaggle_test_result = kaggle_test_dataset.copy()
  kaggle_test_result[CAT_COL_NAME] = kaggle_predict_cat
  kaggle_test_result = kaggle_test_result.drop(columns = [TEXT_COL_NAME])
  kaggle_test_result.head(5)
  kaggle_test_result.to_csv('out.csv')

predict_test('test.csv')
