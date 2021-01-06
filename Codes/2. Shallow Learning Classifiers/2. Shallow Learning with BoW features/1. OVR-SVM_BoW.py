import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import GaussianNB

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def remove_newline_chars(sentence):
	sentence = sentence.replace('\n'," ")
	sentence = sentence.replace('\r'," ")
	sentence = sentence.replace('\t'," ")
	return sentence


'''
lemmatizer=WordNetLemmatizer()

def lemmatize_text(text):
	text = text.replace('\n'," ")
	text = text.replace('\r'," ")
	text = text.replace('\t'," ")
	text = "been had done languages cities mice"
	text = word_tokenize(text)
	lem_list = []
	for word in text:
		lem_list.append(lemmatizer.lemmatize(word))
	lem_str = ' '.join([str(elem.encode("utf-8")) for elem in lem_list])
	print(lem_str)
	return lem_str'''



labeled_data = pd.read_excel("DocumentationSmell_Benchmark_Dataset_Feature.xlsx", encoding='utf-8')


categories = ['Tangled', 'Excessive Structured', 'Fragmented', 'Bloated', 'Lazy']
number_of_class = len(categories)
print(number_of_class)


documentation_text_without_method_prototype_list = []

for index,row in labeled_data.iterrows():
	documentation_text = row['Documentation Text']
	method_prototype = row['Method Prototype']
	documentation_text_without_method_prototype = documentation_text.replace(method_prototype," ")
	documentation_text_without_method_prototype_list.append(documentation_text_without_method_prototype)


labeled_data['Documentation Text'] = documentation_text_without_method_prototype_list

labeled_data['Documentation Text'] = labeled_data['Documentation Text'].apply(remove_newline_chars)
labeled_data['Documentation Text'] = labeled_data['Documentation Text'].str.lower()
#labeled_data['Documentation Text'] = labeled_data['Documentation Text'].apply(cleanHtml)
#labeled_data['Documentation Text'] = labeled_data['Documentation Text'].apply(cleanPunc)
#labeled_data['Documentation Text'] = labeled_data['Documentation Text'].apply(keepAlpha)
labeled_data['Documentation Text'] = labeled_data['Documentation Text'].apply(removeStopWords)
labeled_data['Documentation Text'] = labeled_data['Documentation Text'].apply(stemming)



x = labeled_data[['Documentation Text']].copy()


#print(x[['Documentation Text']].head())


y = labeled_data[categories].copy()


number_of_fold_in_kfold = 5 #5-fold cross valid
mskf = MultilabelStratifiedKFold(n_splits=number_of_fold_in_kfold, shuffle=True, random_state=0)



## using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
print("using binary relevance results:")
print("------------------------------")
print("")


##train test split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, shuffle=True)
# binary relevance using NB ##### better than SVC (For Now)
from sklearn.naive_bayes import GaussianNB
print("using SVM with binary relevance:")
print("------------------------------")
print("")
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
#classifier = BinaryRelevance(GaussianNB())
classifier = BinaryRelevance(SVC(kernel='linear'))





for category in categories:
	print(category)
	print("------------")
	tn = 0
	fp = 0
	fn = 0
	tp = 0
	total_accuracy = 0
	total_precion = 0
	total_recall = 0
	total_f1 = 0
	for train_index, test_index in mskf.split(x, y):
	    #print("TRAIN:", train_index, "TEST:", test_index)
	    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
	    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	    x_train = x_train['Documentation Text'].tolist()
	    x_test = x_test['Documentation Text'].tolist()
	    y_train = y_train.values
	    y_test = y_test.values

	    #n-gram
	    #tfidf = TfidfVectorizer(ngram_range = (1,1), stop_words = 'english')
	    #tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,1), norm='l2')
	    tfidf = CountVectorizer()
	    tfidf.fit(x_train)
	    x_train = tfidf.transform(x_train)
	    x_test = tfidf.transform(x_test)

	    # train
	    #classifier = BinaryRelevance(GaussianNB())
	    classifier.fit(x_train, y_train)

	    # predict
	    predictions = classifier.predict(x_test)

	    y_pred = []
	    for i in predictions:
	    	y_pred.append(list(i.A[0]))

	    #print(y_pred)
	    y_test = y_test.tolist()
	    #print(len(y_test))
	    y_pred_dataframe = pd.DataFrame(y_pred, columns = categories)
	    y_test_dataframe = pd.DataFrame(y_test, columns = categories)
	    #print(len(y_pred_dataframe))
	    this_pred_list = y_pred_dataframe[category].tolist()
	    this_test_list = y_test_dataframe[category].tolist()

	    this_accuracy = accuracy_score(this_test_list,this_pred_list)
	    wt_avg = 'macro'
	    this_precision = precision_score(this_test_list,this_pred_list,average=wt_avg)
	    this_recall = recall_score(this_test_list,this_pred_list,average=wt_avg)
	    this_f1 = f1_score(this_test_list,this_pred_list,average=wt_avg)
	    this_tn, this_fp, this_fn, this_tp = confusion_matrix(this_test_list, this_pred_list).ravel()

	    #print(this_tp)
	    #print(this_fp)
	    #print(this_tn)
	    #print(this_fn)
	    #print("")

	    tn += this_tn
	    tp += this_tp
	    fp += this_fp
	    fn += this_fn
	    total_accuracy += this_accuracy
	    total_precion += this_precision
	    total_recall += this_recall
	    total_f1 += this_f1

	f1 = 1.0 * (2*tp)/(2*tp+fp+fn)
	precision = 1.0 * tp/(tp+fp)
	recall = 1.0 * tp/(tp+fn)
	accuracy = total_accuracy/number_of_fold_in_kfold
	#precision = total_precion/number_of_fold_in_kfold
	#recall= total_recall/number_of_fold_in_kfold
	#f1 = 2.0*precision*recall/(precision+recall)
	print("accuracy:")
	print(accuracy)
	print("f1-score:")
	print(f1)
	print("precision:")
	print(precision)
	print("recall:")
	print(recall)