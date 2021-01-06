import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import hamming_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

data = pd.read_excel("DocumentationSmell_Benchmark_Dataset_Feature.xlsx", encoding='utf-8')

'''feature_list = ['doc_acronyms','doc_broken_urls',
                'doc_jargons','doc_length','doc_misspelled',
                'doc_readability','doc_avg_similarity',
                'edit_distance_between_methodname_and_doc','doc_urls_count',
                'scattered_keyword_count','method_name_count',
                'class_name_count','package_name_count']'''

feature_list = ['doc_acronyms',
                'doc_jargons','doc_length',
                'doc_readability',
                'edit_distance_between_methodname_and_doc','doc_urls_count','method_name_count',
                'class_name_count','package_name_count'] 


#data = pd.merge(labeled_data, feature_data, on=['Id'])
#print(data.head())

categories = ['Tangled', 'Excessive Structured', 'Fragmented', 'Bloated', 'Lazy']
number_of_class = len(categories)
print(number_of_class)

x = data[feature_list]
y = data[categories]



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



new_f1_total = 0

for category in categories:
	print(category)
	print("------------")
	tn = 0
	fp = 0
	fn = 0
	tp = 0
	total_accuracy = 0
	for train_index, test_index in mskf.split(x, y):
	    #print("TRAIN:", train_index, "TEST:", test_index)
	    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
	    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	    x_train = x_train.values
	    x_test = x_test.values
	    y_train = y_train.values
	    y_test = y_test.values

	    ##Robust Scaling of features
	    transformer = RobustScaler().fit(x_train)
	    x_train = transformer.transform(x_train)
	    x_test = transformer.transform(x_test)

	    # train
	    classifier = BinaryRelevance(SVC())
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
	    this_tn, this_fp, this_fn, this_tp = confusion_matrix(this_test_list, this_pred_list).ravel()

	    tn += this_tn
	    tp += this_tp
	    fp += this_fp
	    fn += this_fn
	    total_accuracy += this_accuracy

	f1_score = 1.0 * (2*tp)/(2*tp+fp+fn)
	precision = 1.0 * tp/(tp+fp)
	recall = 1.0 * tp/(tp+fn)
	accuracy = total_accuracy/number_of_fold_in_kfold
	print("accuracy:")
	print(accuracy)
	print("f1-score:")
	print(f1_score)
	print("precision:")
	print(precision)
	print("recall:")
	print(recall)
	new_f1_total += f1_score



new_f1_avg = new_f1_total/number_of_class
#print("avg f1 for all class:")
#print(new_f1_avg)