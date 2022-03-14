import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

import utils

# setup the randoms state
RANDOM_STATE = 545510477

def logistic_regression_pred(X_train, Y_train):
    lr = LogisticRegression()
    lr.fit(X_train,Y_train)
    Y_pred = lr.predict(X_train)
    return Y_pred    

def svm_pred(X_train, Y_train):
    svm = LinearSVC()
    svm.fit(X_train,Y_train)
    Y_pred = svm.predict(X_train)
    return Y_pred

def decisionTree_pred(X_train, Y_train):
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train,Y_train)
    Y_pred = dt.predict(X_train)
    return Y_pred 

#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
    acc = accuracy_score(Y_pred,Y_true)
    auc = roc_auc_score(Y_pred,Y_true)
    precision = average_precision_score(Y_pred,Y_true)
    recall = recall_score(Y_pred,Y_true)
    f1score = f1_score(Y_pred,Y_true)
    return acc,auc,precision,recall,f1score

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	

if __name__ == "__main__":
	main()
	
