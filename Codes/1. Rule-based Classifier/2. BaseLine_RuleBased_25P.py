import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


data = pd.read_excel("DocumentationSmell_Benchmark_Dataset_Feature.xlsx", encoding='utf-8')



feature_list = ['doc_acronyms',
                'doc_jargons','doc_length',
                'doc_readability',
                'edit_distance_between_methodname_and_doc','doc_urls_count',
                'method_name_count',
                'class_name_count','package_name_count']
#categories = ['Tangled', 'Excessive Structured', 'Fragmented', 'Bloated', 'Lazy']
weight_avg = 'macro'
#############################################################################################################################
#feature_list = ['doc_length'] #bloated
#categories = ['Bloated']


#feature_list = ['edit_distance_between_methodname_and_doc'] #lazy
#categories = ['Lazy']


#feature_list = ['method_name_count','class_name_count','package_name_count'] #excess structure
#categories = ['Excessive Structured']


#feature_list = ['doc_urls_count','scattered_keyword_count'] #fragmented
#categories = ['Fragmented']


#feature_list = ['doc_acronyms','doc_jargons','doc_readability'] #tangled
#categories = ['Tangled']
################################################################################################################################################


#print(data.head())

data['total_method_class_package_count'] = data.apply(lambda x: x['method_name_count'] + x['class_name_count'] + x['package_name_count'], axis=1)

#data.to_excel("full_labelled_with_feature.xlsx",index=False)


#number_of_class = len(categories)
#print(number_of_class)

percentile_value = 25
len_threshold = np.percentile(data['doc_length'].tolist(), percentile_value)
edit_distance_threshold = np.percentile(data['edit_distance_between_methodname_and_doc'].tolist(), percentile_value)
readability_threshold = np.percentile(data['doc_readability'].tolist(), percentile_value)
method_class_package_name_threshold = np.percentile(data['package_name_count'].tolist(), percentile_value)
doc_url_count_threshold = np.percentile(data['doc_urls_count'].tolist(), percentile_value)




bloated_true = data['Bloated']
lazy_true = data['Lazy']
tangled_true = data['Tangled']
excess_true = data['Excessive Structured']
fragmented_true = data['Fragmented']


bloated_pred = []
lazy_pred = []
tangled_pred = []
excess_pred = []
fragmented_pred = []


##train test split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, shuffle=True)


for index, row in data.iterrows():
    #bloated
    if row['doc_length'] > len_threshold:
        bloated_pred.append(1)
    else:
        bloated_pred.append(0)
    #lazy
    if row['edit_distance_between_methodname_and_doc'] < edit_distance_threshold:
        lazy_pred.append(1)
    else:
        lazy_pred.append(0)
    #tangled
    if row['doc_readability'] < readability_threshold:
        tangled_pred.append(1)
    else:
        tangled_pred.append(0)
    #excess
    #if row['total_method_class_package_count'] > method_class_package_name_threshold:
    if row['package_name_count'] > method_class_package_name_threshold:
        excess_pred.append(1)
    else:
        excess_pred.append(0)
    #fragmented
    if row['doc_urls_count'] > doc_url_count_threshold:
        fragmented_pred.append(1)
    else:
        fragmented_pred.append(0)



true_df_values = pd.DataFrame(list(zip(bloated_true, lazy_true,tangled_true,excess_true,fragmented_true))).values
pred_df_values = pd.DataFrame(list(zip(bloated_pred, lazy_pred,tangled_pred,excess_pred,fragmented_pred))).values

print("Baseline Result:")
print("-----------------")
print("")


print("class wise result for baseline model:")
print("--------------------------------------")



print("Bloated:")
print("----------")
tn, fp, fn, tp = confusion_matrix(bloated_true, bloated_pred).ravel()
acc_bloated = accuracy_score(bloated_true, bloated_pred)
pre = 1.0 * tp/(tp+fp)
#pre = precision_score(bloated_true, bloated_pred, average=weight_avg)
rec = recall_score(bloated_true, bloated_pred, average=weight_avg)
f1 = 2.0 * ((pre * rec)/(pre+rec))

#tn, fp, fn, tp = confusion_matrix(bloated_true, bloated_pred).ravel()
#pre = 1.0 * tp/(tp+fp)
#rec = 1.0 * tp/(tp+fn)
#f1 = 1.0 * (2*tp)/(2*tp+fp+fn)

print("acc: ")
print(acc_bloated)
print("precision: ")
print(pre)
print("Recall:")
print(rec)
print("F1:")
print(f1)
print("")




print("Lazy:")
print("----------")
tn, fp, fn, tp = confusion_matrix(lazy_true, lazy_pred).ravel()
acc_lazy = accuracy_score(lazy_true, lazy_pred)
pre = 1.0 * tp/(tp+fp)
#pre = precision_score(lazy_true, lazy_pred, average=weight_avg)
rec = recall_score(lazy_true, lazy_pred, average=weight_avg)
f1 = 2.0 * ((pre * rec)/(pre+rec))


#tn, fp, fn, tp = confusion_matrix(lazy_true, lazy_pred).ravel()
#pre = 1.0 * tp/(tp+fp)
#rec = 1.0 * tp/(tp+fn)
#f1 = 1.0 * (2*tp)/(2*tp+fp+fn)

print("acc: ")
print(acc_lazy)
print("precision: ")
print(pre)
print("Recall:")
print(rec)
print("F1:")
print(f1)
print("")




print("Tangled:")
print("----------")
acc_tangled = accuracy_score(tangled_true, tangled_pred)
pre = precision_score(tangled_true, tangled_pred, average=weight_avg)
rec = recall_score(tangled_true, tangled_pred, average=weight_avg)
f1 = 2.0 * ((pre * rec)/(pre+rec))

tn, fp, fn, tp = confusion_matrix(tangled_true, tangled_pred).ravel()
pre = 1.0 * tp/(tp+fp)
rec = 1.0 * tp/(tp+fn)
f1 = 1.0 * (2*tp)/(2*tp+fp+fn)

print("acc: ")
print(acc_tangled)
print("precision: ")
print(pre)
print("Recall:")
print(rec)
print("F1:")
print(f1)
print("")




print("Fragmented:")
print("----------")
acc_fragmented = accuracy_score(fragmented_true, fragmented_pred)
pre = precision_score(fragmented_true, fragmented_pred, average=weight_avg)
rec = recall_score(fragmented_true, fragmented_pred, average=weight_avg)
f1 = 2.0 * ((pre * rec)/(pre+rec))

tn, fp, fn, tp = confusion_matrix(fragmented_true, fragmented_pred).ravel()
pre = 1.0 * tp/(tp+fp)
rec = 1.0 * tp/(tp+fn)
f1 = 1.0 * (2*tp)/(2*tp+fp+fn)


print("acc: ")
print(acc_fragmented)
print("precision: ")
print(pre)
print("Recall:")
print(rec)
print("F1:")
print(f1)
print("")

print("Excessive Stuct:")
print("----------")
acc_excess = accuracy_score(excess_true, excess_pred)
pre = precision_score(excess_true, excess_pred, average=weight_avg)
rec = recall_score(excess_true, excess_pred, average=weight_avg)
f1 = 2.0 * ((pre * rec)/(pre+rec))

tn, fp, fn, tp = confusion_matrix(excess_true, excess_pred).ravel()
pre = 1.0 * tp/(tp+fp)
rec = 1.0 * tp/(tp+fn)
f1 = 1.0 * (2*tp)/(2*tp+fp+fn)

print("acc: ")
print(acc_excess)
print("precision: ")
print(pre)
print("Recall:")
print(rec)
print("F1:")
print(f1)
print("")
